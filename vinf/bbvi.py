import torch
from torch.optim import Adam
import numpy as np
from scipy.special import polygamma
from tqdm import trange
from vinf.utils import get_optimizer, symmetrize


def bbvirp(vars, sample_reparam, normalize_sample, joint, proposal, step_size=0.1,
           n_iter=10000, n_samples=1, track_grad_var=False, track_grad_var_freq=100,
           grad_var_samples=1000, track_elbo=False, n_samples_elbo=50, opt='Adam',
           vstep_sizes=None, test_joint=None):
    n_samples *= 3
    vstep_sizes = dict() if vstep_sizes is None else vstep_sizes
    O = get_optimizer(opt)
    opt = O([{'params': vars[k], 'lr': vstep_sizes.get(k, step_size)}
             for k in vars if vars[k].requires_grad], lr=step_size)
    hist = {k: [] for k in vars if vars[k].requires_grad}
    grad_vars = list()
    elbos = list()
    if test_joint:
        test_losses = list()
    S = n_samples
    rs = 0
    psd_const = 'S' if 'S' in vars else ('W' if 'W' in vars else None)
    for it in trange(n_iter):
        if track_grad_var and (it % track_grad_var_freq) == 0:
            grad_vars.append(bbvi_grad_var(vars, joint, proposal, grad_var_samples))
        opt.zero_grad()
        # gradient of entropy of proposal
        dist_prop_grad_df = proposal({k: vars[k].detach() if 'df' not in k else vars[k] for k in vars})  # BBVI D df
        dist_prop_grad_rp = proposal({k: vars[k].detach() if 'df' in k else vars[k] for k in vars})  # reparam D Rest
        loss = - dist_prop_grad_rp.entropy()

        # sampling
        det_vars = {k: vars[k].detach() for k in vars}
        dist_prop = proposal(det_vars)

        if track_elbo:
            samples = dist_prop.sample((n_samples_elbo,))
            elbo = torch.zeros(1, dtype=torch.float64)
            for i in range(n_samples_elbo):
                xent = joint.log_prob(samples[i])
                ent = dist_prop.log_prob(samples[i])
                elbo += xent - ent
            elbos.append((elbo / n_samples_elbo).item())
            if test_joint:
                test_loss = 0
                for i in range(n_samples_elbo):
                    test_loss += test_joint.llh(samples[i]).item()
                test_losses.append(test_loss)

        samples = dist_prop.sample((S,))

        # bbvi grad
        for i in range(S):
            coeff = joint.log_prob(samples[i]) - dist_prop.log_prob(samples[i])
            score = dist_prop_grad_df.log_prob(samples[i])
            loss -= score * coeff / S

        # reparam grad
        for i in range(S):
            loss -= joint.log_prob(sample_reparam(normalize_sample(samples[i], det_vars), vars)) / S

        loss.backward()
        if psd_const in vars:
            prev = vars[psd_const].data.numpy().copy()

        symmetrize(vars)

        opt.step()

        if 'D' in vars:
            vars['D'].data.abs_()

        if psd_const in vars:
            try:
                new = vars[psd_const].data.numpy().copy()
                np.linalg.cholesky(new)
            except:
                rs += 1
                vars[psd_const].data.add_(torch.from_numpy(prev)).sub_(torch.from_numpy(new))
        for k in hist:
            try:
                hist[k].append(vars[k].item())
            except:
                hist[k].append(vars[k].data.numpy().flatten())
    for k in hist:
        hist[k] = np.array(hist[k])
    if rs:
        print(rs, 'restarts')
    if test_joint:
        return hist, elbos, grad_vars, test_losses
    return hist, elbos, grad_vars


def bbvi_grad_var(vars, joint, proposal, n_samples, true_grad=None):
    if 'df' not in vars:
        return None
    grad_samples = list()
    vars = {k: (vars[k] if 'df' in k else vars[k].detach()) for k in vars}
    opt = Adam([vars['df']])  # just used for 0-grad

    dist_prop = proposal({k: vars[k].detach() for k in vars})
    samples = dist_prop.sample((2*n_samples,))

    for i in trange(n_samples):
        opt.zero_grad()
        loss = torch.zeros(1, dtype=torch.float64)
        dist_prop_grad = proposal(vars)
        for j in range(2):
            coeff = joint.log_prob(samples[2*i+j]) - dist_prop.log_prob(samples[2*i+j])
            score = dist_prop_grad.log_prob(samples[2*i+j])
            loss -= score * coeff / 2
        loss.backward()
        grad_samples.append(vars['df'].grad.item())
    grad_samples = np.array(grad_samples)
    if true_grad is not None:
        bias_squared = (np.mean(grad_samples) - true_grad) ** 2
        variance = np.mean((grad_samples - np.mean(grad_samples)) ** 2)
        return bias_squared, variance
    return np.mean(grad_samples), np.var(grad_samples)


def bbvi_grep_var(vars, joint, proposal, n_samples, true_grad=None):
    grad_samples = list()
    vars = {k: (vars[k] if 'df' in k else vars[k].detach()) for k in vars}

    dist_prop = proposal({k: vars[k].detach() for k in vars})
    samples = dist_prop.sample((n_samples,))
    alpha, beta = vars['df'].item(), vars['rate'].item()
    psi, psi_one, psi_two = polygamma([0, 1, 2], alpha)

    if vars['df'].grad is not None:
        vars['df'].grad.data.zero_()
    prop_grad = proposal(vars)
    loss_ent = prop_grad.entropy()
    loss_ent.backward()
    dH_ddf = vars['df'].grad.item()
    gcorrs, greps = list(), list()
    for i in trange(n_samples):
        # GREP
        z = samples[i].item()
        g = torch.tensor([z], dtype=torch.double, requires_grad=True)
        fz = joint.log_prob(g)
        fz.backward()
        grad_fz = g.grad.item()
        g.grad.data.zero_()
        fz = fz.item()
        eps = (np.log(z) - psi + np.log(beta)) / np.sqrt(psi_one)
        rep_factor = (0.5 * psi_two / np.sqrt(psi_one)) * eps + psi_one
        grep = grad_fz * rep_factor * z

        # GCORR
        lt = psi_two / (2 * np.sqrt(psi_one))
        dL_da = (0.5 * psi_two / psi_one + alpha * psi_one
                 + eps * (np.sqrt(psi_one) + alpha * lt)
                 - (lt * eps + psi_one) * np.exp(np.sqrt(psi_one) * eps + psi))
        gcorr = fz * dL_da
        grad_samples.append(grep + gcorr + dH_ddf)
        gcorrs.append(gcorr)
        greps.append(grep)
    grad_samples = - np.array(grad_samples)
    if true_grad is not None:
        bias_squared = (np.mean(grad_samples) - true_grad) ** 2
        variance = np.mean((grad_samples - np.mean(grad_samples)) ** 2)
        return bias_squared, variance
    return np.mean(grad_samples), np.var(grad_samples)
