import torch
from torch.optim import Adam
import numpy as np
from sacred import Experiment
from tqdm import trange
from copy import deepcopy
import pickle

from vinf.bbvi import bbvi_grad_var, bbvi_grep_var
from vinf.vind import vind_grad_var
from vinf.joints import GammaNormal
from vinf.proposals import Gamma
from vinf.exact_losses import GammaLoss
from vinf.datasets import GammaNormal as GN_ds


ex = Experiment()


@ex.config
def configuration():
    mu = 5
    tau = 10
    data_size = 500
    df_prior = 30
    rate_prior = 10
    df_init = 500  # or 'post'
    rate_init = 'post'  # or 'post'
    n_iter = 400
    n_samples_mse = 5000
    vind_epss = [1, 5, 10, 100]
    step_size = 1


@ex.capture
def init(mu, tau, data_size, df_prior, rate_prior, df_init, rate_init, _seed, _config):
    ds = GN_ds(mu, tau)
    d = ds.generate(data_size, seed=_seed)
    rate_post = rate_prior + np.sum((d.numpy() - mu)**2) / 2
    df_post = data_size / 2 + df_prior
    _config.update({'rate_post': rate_post,
                    'df_post': df_post})
    rate_init = rate_post if rate_init == 'post' else rate_init
    df_init = df_post if df_init == 'post' else df_init
    vars_ = {'df': torch.tensor([df_init], dtype=torch.float64),
             'rate': torch.tensor([rate_init], dtype=torch.float64)}
    vars_['df'].requires_grad = True
    df_post = torch.tensor([df_post], dtype=torch.float64)
    rate_post = torch.tensor([rate_post], dtype=torch.float64)
    joint = GammaNormal(d, mu, df_prior, rate_prior)
    return vars_, joint, df_post, rate_post


@ex.automain
def gn_experiment(step_size, n_iter, n_samples_mse, vind_epss, _run):
    vars_, joint, df_post, rate_post = init()
    opt = Adam([vars_['df']], lr=step_size)
    elbo_loss = GammaLoss(df_post, rate_post)
    biases1 = {'BBVI': list(), 'GREP': list()}
    biases2 = {'VIND-{eps}'.format(eps=eps): list() for eps in vind_epss}
    biases = {**biases1, **biases2}
    variances = deepcopy(biases)
    metrics = {'KL': list(), 'EPost': list(), 'EApprox': list(), 'df': list()}
    iterations = list(range(1, n_iter+1))
    for _ in trange(n_iter):
        opt.zero_grad()
        loss = elbo_loss(vars_)
        loss.backward()
        true_grad = vars_['df'].grad.item()
        varsr = deepcopy(vars_)
        bias_squared, variance = bbvi_grad_var(varsr, joint, Gamma, n_samples_mse, true_grad)
        biases['BBVI'].append(bias_squared)
        variances['BBVI'].append(variance)
        varsr = deepcopy(vars_)
        bias_squared, variance = bbvi_grep_var(varsr, joint, Gamma, n_samples_mse, true_grad)
        biases['GREP'].append(bias_squared)
        variances['GREP'].append(variance)
        for eps in vind_epss:
            varsr = deepcopy(vars_)
            bias_squared, variance = vind_grad_var(varsr, joint, Gamma, eps, n_samples_mse, true_grad)
            biases['VIND-{eps}'.format(eps=eps)].append(bias_squared)
            variances['VIND-{eps}'.format(eps=eps)].append(variance)
        metrics['KL'].append(loss.item())
        metrics['EPost'].append(elbo_loss.true_dist.mean.item())
        metrics['EApprox'].append(vars_['df'].item() / vars_['rate'].item())
        metrics['df'].append(vars_['df'].item())
        opt.step()
    result = {'biases': biases, 'variances': variances, 'metrics': metrics, 'iterations': iterations}
    with open('mse_grad_gamma.pkl', 'wb') as f:
        pickle.dump(result, f)
