import torch
import numpy as np
import scipy as sp
from sacred import Experiment
from copy import deepcopy

from vinf.bbvi import bbvirp
from vinf.vind import double_vind
from vinf.joints import WishartGammaNormalStudent
from vinf.proposals import WishartGammaNormal
from vinf.datasets import DowJones
from vinf.reparams import WishartStudentNormalReparam as WSR


ex = Experiment()


@ex.config
def configuration():
    d = 29
    W_df_prior = 100
    W_init_factor = 1
    W_df_init = 100
    G_df_prior = 5
    G_rate_prior = 1  # mode at 4
    G_df_init = 10
    G_rate_init = 10
    alpha_prior = 0.3
    alpha_init = 0.3
    prior_correlation = True

    methods = {
        'BBVI-RP': {'method': 'bbvirp', 'step_size': 0.01},
        'VIND-1-': {'method': 'vind', 'step_size': 0.01, 'eps': {'W_df': 2 * d, 'G_df': 1}, 'vind_on_ent': False},
        'VIND-2-1': {'method': 'vind', 'step_size': 0.01, 'eps': {'W_df': 4 * d, 'G_df': 1}, 'vind_on_ent': False},
        'VIND-2-2': {'method': 'vind', 'step_size': 0.01, 'eps': {'W_df': 4 * d, 'G_df': 2}, 'vind_on_ent': False},
        'VIND-1-2': {'method': 'vind', 'step_size': 0.01, 'eps': {'W_df': 2 * d, 'G_df': 2}, 'vind_on_ent': False},
        'BBVI-RP-wodf': {'method': 'bbvirp-wodf', 'step_size': 0.01}
    }

    method_conf = {
        'n_iter': 5000,
        'n_samples': 1,
        'track_elbo': True,
        'n_samples_elbo': 100,
        'opt': 'Adam',
        'vstep_sizes': {
            'W_df': 0.1,
            'W': 0.00001,
            'G_df': 0.1,
            'rate': 0.1,
            'loc': 0.1,
            'alpha': 0.1
        },
        'track_grad_var': False,
        'track_grad_var_freq': 100,
        'grad_var_samples': 1000
    }


@ex.capture
def init(W_df_init, W_df_prior, W_init_factor, prior_correlation, alpha_init, alpha_prior,
         G_df_init, G_rate_init, G_df_prior, G_rate_prior, d, _config):
    factor = 1e3
    data, test_data = DowJones(sub_mean=False, dim=d, factor=factor)
    # 0.0008 is variance of stocks before train and test data estimated
    # we scale by factor ** 2 due to var(cx) = c^2 var(x)
    prior_variance = 0.0008 * factor ** 2

    Cemp = np.cov(data.numpy().T)
    Pemp = np.linalg.pinv(Cemp)  # use Pemp instead of W_p_unnorm

    # new
    W_p_unnorm = 1 / prior_variance * np.eye(d)
    W_prior = W_p_unnorm / W_df_prior
    joint = WishartGammaNormalStudent(data, 0, alpha_prior, W_df_prior, W_prior, G_df_prior, G_rate_prior)
    test_joint = WishartGammaNormalStudent(test_data, 0, alpha_prior, W_df_prior, W_prior, G_df_prior, G_rate_prior)
    W_init = W_init_factor * sp.linalg.sqrtm(Pemp / W_df_init)
    W_init = torch.from_numpy(W_init.astype(np.float64))

    W_init.requires_grad = True
    # rate init
    G_rate_init = torch.tensor([G_rate_init], dtype=torch.float64)
    G_rate_init.requires_grad = True
    # w and g df inits
    W_df_init = torch.tensor([W_df_init], dtype=torch.float64)
    G_df_init = torch.tensor([G_df_init], dtype=torch.float64)
    W_df_init.requires_grad = True
    G_df_init.requires_grad = True
    # loc and alpha
    loc_init = torch.zeros(d, dtype=torch.double)
    alpha_init = torch.tensor([alpha_init], dtype=torch.double)
    loc_init.requires_grad = True
    alpha_init.requires_grad = True
    vars_ = {
        'W_df': W_df_init,
        'W': W_init,
        'G_df': G_df_init,
        'rate': G_rate_init,
        'loc': loc_init,
        'alpha': alpha_init
    }
    return vars_, joint, test_joint


@ex.automain
def wn_experiment(methods, method_conf, _run, _log):
    import pickle
    vars_, joint, test_joint = init()
    result = {name: dict() for name in methods}
    for name, conf in methods.items():
        varsr = deepcopy(vars_)
        _log.info('Now Running ' + name)
        _log.info('With configuration: ' + str(conf))
        if conf['method'] == 'bbvirp':
            hist, elbo, _, testloss = bbvirp(varsr, WSR.sample_reparam, WSR.normalize_sample, joint,
                                             WishartGammaNormal, conf['step_size'], **method_conf,
                                             test_joint=test_joint)
        elif conf['method'] == 'bbvirp-wodf':
            varsr['W_df'].requires_grad = False
            hist, elbo, _, testloss = bbvirp(varsr, WSR.sample_reparam, WSR.normalize_sample, joint,
                                             WishartGammaNormal, conf['step_size'], **method_conf,
                                             test_joint=test_joint)
        elif conf['method'] == 'vind':
            hist, elbo, fgrad, testloss = double_vind(varsr, WSR.sample_reparam, WSR.normalize_sample, joint,
                                                      WishartGammaNormal, conf['eps'], conf['step_size'], **method_conf,
                                                      test_joint=test_joint)
        else:
            raise ValueError(conf['method'] + ' is not a valid method')
        hist['EGamma'] = [df/rate for df, rate in zip(hist['G_df'], hist['rate'])]
        result[name]['elbo'] = elbo
        result[name]['hist'] = hist
        result[name]['testloss'] = testloss
    fname = 'wishart_student_normal'
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(result, f)
