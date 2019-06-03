import torch
from torch.optim import Adam
import numpy as np
from tqdm import trange
from copy import copy
from vinf.utils import get_optimizer, symmetrize


EPS = 1e-7
INC_EPS = 0.2


def vind(vars, sample_reparam, normalize_sample, joint, proposal, eps=1., step_size=0.1,
         n_iter=10000, n_samples=1, track_grad_var=False, track_grad_var_freq=100,
         grad_var_samples=1000, track_elbo=False, n_samples_elbo=50, opt='Adam',
         vind_on_ent=False, vstep_sizes=None, test_joint=None):
    vstep_sizes = dict() if vstep_sizes is None else vstep_sizes
    epss = list()
    original_eps = eps
    O = get_optimizer(opt)
    opt = O([{'params': vars[k], 'lr': vstep_sizes.get(k, step_size)}
             for k in vars if vars[k].requires_grad])
    hist = {k: [] for k in vars if vars[k].requires_grad}
    grad_vars = list()
    elbos = list()
    S = n_samples
    rs = 0
    if test_joint:
        test_losses = list()
    psd_const = 'S' if 'S' in vars else ('W' if 'W' in vars else None)
    for it in trange(n_iter):
        if track_grad_var and (it % track_grad_var_freq) == 0:
            grad_vars.append(vind_grad_var(vars, joint, proposal, eps, grad_var_samples))

        opt.zero_grad()
        # gradient of entropy of proposal
        if vind_on_ent:
            dist_prop_grad = proposal({k: vars[k].detach() if k == 'df' else vars[k] for k in vars})  # reparam D Rest
        else:
            dist_prop_grad = proposal(vars)
        loss = - dist_prop_grad.entropy()

        # sampling
        det_vars = {k: vars[k].detach() for k in vars}
        dist_prop = proposal(det_vars)

        if track_elbo:
            samples = dist_prop.sample((n_samples_elbo,))
            elbo = torch.zeros(1, dtype=torch.float64)
            for i in range(n_samples_elbo):
                elbo += joint.log_prob(samples[i]) - dist_prop.log_prob(samples[i])
            elbos.append((elbo / n_samples_elbo).item())
            if test_joint:
                test_loss = 0
                for i in range(n_samples_elbo):
                    test_loss += test_joint.llh(samples[i]).item()
                test_losses.append(test_loss)
        if original_eps != eps:
            while (vars['df'].item() - eps/2 >= INC_EPS) and eps < original_eps:
                eps = eps * 2
        while vars['df'].item() - eps/2 <= EPS:
            eps = eps / 2
        samples_neg, samples, samples_pos = dist_prop.sample_pos_neg((S,), eps=eps)

        # vinf grad
        df_grad = torch.zeros(1, dtype=torch.float64)
        for i in range(S):
            if vind_on_ent:
                coeff_neg = joint.log_prob(samples_neg[i]) - dist_prop.log_prob(samples_neg[i])
                coeff_pos = joint.log_prob(samples_pos[i]) - dist_prop.log_prob(samples_pos[i])
            else:
                coeff_neg = joint.log_prob(samples_neg[i])
                coeff_pos = joint.log_prob(samples_pos[i])
            df_grad += (coeff_pos - coeff_neg)
        df_grad = - df_grad / (eps * S)
        if torch.isinf(df_grad):
            eps = eps / 2
            df_grad = torch.zeros(1, dtype=torch.float64)

        # reparam grad
        for i in range(S):
            loss -= joint.log_prob(sample_reparam(normalize_sample(samples[i], det_vars), vars)) / S

        loss.backward()
        if vind_on_ent:
            vars['df'].grad = df_grad
        else:
            vars['df'].grad += df_grad

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
        epss.append(copy(eps))
    for k in hist:
        hist[k] = np.array(hist[k])
    if rs:
        print(rs, 'restarts')
    if test_joint:
        return hist, elbos, grad_vars, epss, test_losses
    return hist, elbos, grad_vars, epss


def vind_grad_var(vars, joint, proposal, eps, n_samples, true_grad=None):
    if 'df' not in vars:
        return None
    grad_samples = list()
    while vars['df'].item() - eps / 2 <= EPS:
        eps = eps / 2
    vars = {k: (vars[k] if k == 'df' else vars[k].detach()) for k in vars}
    opt = Adam([vars['df']])
    opt.zero_grad()

    # sampling
    det_vars = {k: vars[k].detach() for k in vars}
    dist_prop = proposal(det_vars)
    samples_neg, samples, samples_pos = dist_prop.sample_pos_neg((n_samples,), eps=eps)

    # H gradient
    dist_prop_grad = proposal(vars)
    loss = - dist_prop_grad.entropy()
    loss.backward()
    dH_ddf = vars['df'].grad
    for i in trange(n_samples):
        coeff_neg = joint.log_prob(samples_neg[i])
        coeff_pos = joint.log_prob(samples_pos[i])
        dC_ddf = (coeff_neg - coeff_pos) / eps
        if not torch.isfinite(dC_ddf):
            print('non-finite gradient.')
            return vind_grad_var(vars, joint, proposal, eps/2, n_samples)
        dE_ddf = dH_ddf + dC_ddf
        grad_samples.append(dE_ddf.item())
    grad_samples = np.array(grad_samples)
    if true_grad is not None:
        bias_squared = (np.mean(grad_samples) - true_grad) ** 2
        variance = np.mean((grad_samples - np.mean(grad_samples)) **2)
        return bias_squared, variance
    return np.mean(grad_samples), np.var(grad_samples)


def double_vind(vars, sample_reparam, normalize_sample, joint, proposal, eps=None, step_size=0.1,
                n_iter=10000, n_samples=1, track_elbo=False, n_samples_elbo=50, opt='Adam',
                vind_on_ent=False, vstep_sizes=None, track_grad_var=False, track_grad_var_freq=100,
                grad_var_samples=1000, test_joint=None):
    vstep_sizes = dict() if vstep_sizes is None else vstep_sizes
    O = get_optimizer(opt)
    opt = O([{'params': vars[k], 'lr': vstep_sizes.get(k, step_size)}
             for k in vars if vars[k].requires_grad], lr=step_size)
    hist = {k: [] for k in vars if vars[k].requires_grad}
    elbos = list()
    if test_joint:
        test_losses = list()
    S = n_samples
    df_params = [k for k in vars if 'df' in k]
    grad_vars = list()
    for it in trange(n_iter):
        if track_grad_var and (it % track_grad_var_freq) == 0:
            grad_vars.append(double_grad_var(vars, joint, proposal, eps, grad_var_samples,
                                             sample_reparam, normalize_sample))

        opt.zero_grad()
        # gradient of entropy of proposal
        if vind_on_ent:
            dist_prop_grad = proposal({k: vars[k].detach() if 'df' in k else vars[k] for k in vars})
            # reparam D Rest
        else:
            dist_prop_grad = proposal(vars)
        loss = - dist_prop_grad.entropy()

        # sampling
        det_vars = {k: vars[k].detach() for k in vars}
        dist_prop = proposal(det_vars)

        if track_elbo:
            samples = dist_prop.sample((n_samples_elbo,))
            elbo = torch.zeros(1, dtype=torch.float64)
            for i in range(n_samples_elbo):
                elbo += joint.log_prob(samples[i]) - dist_prop.log_prob(samples[i])
            elbos.append((elbo / n_samples_elbo).item())
            if test_joint:
                test_loss = 0
                for i in range(n_samples_elbo):
                    test_loss += test_joint.llh(samples[i]).item()
                test_losses.append(test_loss)

        samples_neg, samples, samples_pos = dist_prop.sample_mult_eps((S,), eps=eps)

        # vinf grad
        df_grads = dict()
        for k in df_params:
            df_grad = torch.zeros(1, dtype=torch.float64)
            for i in range(S):
                if vind_on_ent:
                    coeff_neg = joint.log_prob(samples_neg[k][i]) - dist_prop.log_prob(samples_neg[k][i])
                    coeff_pos = joint.log_prob(samples_pos[k][i]) - dist_prop.log_prob(samples_pos[k][i])
                else:
                    coeff_neg = joint.log_prob(samples_neg[k][i])
                    coeff_pos = joint.log_prob(samples_pos[k][i])
                df_grad += (coeff_pos - coeff_neg)
            df_grads[k] = - df_grad / (eps[k] * S)

        # reparam grad
        for i in range(S):
            loss -= joint.log_prob(sample_reparam(normalize_sample(samples[i], det_vars), vars)) / S

        loss.backward()
        for k in df_params:
            if vind_on_ent:
                vars[k].grad = df_grads[k]
            else:
                vars[k].grad += df_grads[k]

        opt.step()

        for k in hist:
            try:
                hist[k].append(vars[k].item())
            except:
                hist[k].append(vars[k].data.numpy().flatten())
    for k in hist:
        hist[k] = np.array(hist[k])
    if test_joint:
        return hist, elbos, grad_vars, test_losses
    return hist, elbos, grad_vars


def double_grad_var(vars, joint, proposal, eps, n_samples, reparam, normalize):
    res = dict()
    mus = dict()
    grad_samples = list()
    grad_samples_uncoupled = list()
    vars = {k: (vars[k] if k == 'W_df' or k == 'W' else vars[k].detach()) for k in vars}
    opt = Adam([vars['W_df'], vars['W']], lr=0.0)
    opt.zero_grad()

    # sampling
    det_vars = {k: vars[k].detach() for k in vars}
    dist_prop = proposal(det_vars)
    samples_neg, samples, samples_pos = dist_prop.sample_mult_eps((2*n_samples,), eps=eps)

    # H gradient
    dist_prop_grad_df = proposal({k: vars[k].detach() if k == 'W' else vars[k] for k in vars})
    loss = - dist_prop_grad_df.entropy()
    loss.backward()
    dH_ddf = vars['W_df'].grad
    for i in trange(n_samples):
        coeff_neg = joint.log_prob(samples_neg['W_df'][i])
        coeff_pos = joint.log_prob(samples_pos['W_df'][i])
        dC_ddf = (coeff_neg - coeff_pos) / eps['W_df']
        dE_ddf = dH_ddf + dC_ddf
        grad_samples.append(dE_ddf.item())
        # non-coupled
        coeff_pos = joint.log_prob(samples_pos['W_df'][(i+1) % n_samples])
        dC_ddf_uncoupled = (coeff_neg - coeff_pos) / eps['W_df']
        dE_ddf_uncoupled = dH_ddf + dC_ddf_uncoupled
        grad_samples_uncoupled.append(dE_ddf_uncoupled.item())
    res['df_VIND'] = np.var(grad_samples)
    mus['df_VIND'] = np.mean(grad_samples)
    res['df_UNC'] = np.var(grad_samples_uncoupled)
    mus['df_UNC'] = np.mean(grad_samples_uncoupled)
    opt.zero_grad()
    opt.step()

    dist_prop_grad_W = proposal({k: vars[k].detach() if k == 'W_df' else vars[k] for k in vars})
    lossb = dist_prop_grad_W.entropy()
    lossb.backward()
    dH_ddW = vars['W'].grad[0, 0].item()
    opt.step()
    opt.zero_grad()
    grad_samples_df = list()
    grad_samples_W = list()
    for i in trange(n_samples):
        opt.zero_grad()
        loss = torch.zeros(1, dtype=torch.float64)
        dist_prop_grad_df = proposal({k: vars[k].detach() if k == 'W' else vars[k] for k in vars})
        for j in range(2):
            coeff = joint.log_prob(samples[i*2+j], blackwell=True) - dist_prop.log_prob(samples[i*2+j], blackwell=True)
            score = dist_prop_grad_df.log_prob(samples[i*2+j], blackwell=True)
            loss -= score * coeff / 2
            loss -= joint.log_prob(reparam(normalize(samples[i*2+j], det_vars), vars)) / 2
        loss.backward()
        symmetrize(vars)
        grad_samples_df.append(vars['W_df'].grad.item())
        grad_samples_W.append(vars['W'].grad[0, 0].item() + dH_ddW)
    res['df_BBVI'] = np.var(grad_samples_df)
    res['W'] = np.var(grad_samples_W)
    mus['df_BBVI'] = np.mean(grad_samples_df)
    print('---results---')
    print(mus)
    print(res)
    return res
