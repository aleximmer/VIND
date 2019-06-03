import torch
import numpy as np
from sacred import Experiment
from copy import deepcopy

from vinf.bbvi import bbvirp
from vinf.vind import vind
from vinf.joints import MVNGammaLinRegIndep
from vinf.proposals import InGammaMVN
from vinf.datasets import BostonHousing
from vinf.reparams import LinRegIndReparam


ex = Experiment()


@ex.config
def configuration():
    df_prior = 5
    rate_prior = 5
    df_init = 200
    rate_init = 50
    S_prior_factor = 1
    S_init_factor = 1
    only_df = False

    methods = {
        'VIND-0_1-Ent': {'method': 'vind', 'step_size': 1, 'eps': 2},
        'VIND-0_5-Ent': {'method': 'vind', 'step_size': 1, 'eps': 2},
        'VIND-1-Ent': {'method': 'vind', 'step_size': 1, 'eps': 2},
        'VIND-2-Ent': {'method': 'vind', 'step_size': 1, 'eps': 2},
        'BBVI-RP': {'method': 'bbvirp', 'step_size': 1}
    }

    method_conf = {
        'n_iter': 2000,
        'n_samples': 1,
        'track_grad_var': False,
        'track_grad_var_freq': 200,
        'grad_var_samples': 1000,
        'track_elbo': True,
        'n_samples_elbo': 10,
        'opt': 'Adam',
        'vstep_sizes': {
            'D': 0.1,
            'loc': 1,
            'df': 5,
            'rate': 5
        }
    }

    # data set
    decorrelate = True


@ex.capture
def init(df_init, rate_init, S_init_factor, df_prior, rate_prior, S_prior_factor,
         only_df, decorrelate):
    data = BostonHousing(decorrelate)
    dim = data[0].shape[1]
    jdist = MVNGammaLinRegIndep
    joint = jdist(data[:2], S_prior_factor, df_prior, rate_prior)
    test_joint = jdist(data[2:], S_prior_factor, df_prior, rate_prior)

    mu_init = np.zeros(dim).astype(np.float64)
    S_init = S_init_factor * np.eye(dim).astype(np.float64)
    loc = torch.from_numpy(mu_init)
    df = torch.tensor([df_init], dtype=torch.float64)
    rate = torch.tensor([rate_init], dtype=torch.float64)
    loc.requires_grad = True
    df.requires_grad = True
    rate.requires_grad = True

    D = torch.from_numpy(np.diag(S_init))
    D.requires_grad = not only_df
    vars_ = {'D': D, 'loc': loc, 'df': df, 'rate': rate}

    return vars_, joint, test_joint


@ex.automain
def lir_experiment(methods, method_conf, _run, _log):
    import pickle
    vars_, joint, test_joint = init()
    result = {name: {'elbo': list(), 'testloss': list()} for name in methods}
    for name, conf in methods.items():
        varsr = deepcopy(vars_)
        prop = InGammaMVN
        R = LinRegIndReparam
        if conf['method'] == 'bbvirp':
            hist, elbo, grad_var, testloss = bbvirp(varsr, R.sample_reparam, R.normalize_sample, joint,
                                          prop, conf['step_size'], test_joint=test_joint, **method_conf)
        elif conf['method'] == 'vind':
            hist, elbo, grad_var, es, testloss = vind(varsr, R.sample_reparam, R.normalize_sample, joint, prop,
                                                      conf['eps'], conf['step_size'], test_joint=test_joint,
                                                      vind_on_ent=conf['vind_on_ent'],
                                                      **method_conf)
        else:
            raise ValueError(conf['method'] + ' is not a valid method')
        result[name]['elbo'] = elbo
        result[name]['testloss'] = testloss
    with open('linear_regression.pkl', 'wb') as f:
        pickle.dump(result, f)
