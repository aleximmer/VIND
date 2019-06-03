import numpy as np
import scipy as sp
import scipy.special
from torch.distributions import kl_divergence, Gamma


class GammaLoss:

    def __init__(self, posterior_df, posterior_rate):
        self.alpha = posterior_df
        self.beta = posterior_rate
        self.true_dist = Gamma(concentration=posterior_df, rate=posterior_rate)

    def __call__(self, vars):
        approx_dist = Gamma(concentration=vars['df'], rate=vars['rate'])
        return kl_divergence(approx_dist, self.true_dist)

    def sp(self, vars):
        # KL divergence of true FROM proposal
        alpha_ = vars['df'].item()
        beta_ = vars['rate'].item()
        kl_div = (alpha_ - self.alpha) * sp.special.digamma(alpha_) - sp.special.gammaln(alpha_) \
                 + sp.special.gammaln(self.alpha) + self.alpha * (np.log(beta_) - np.log(self.beta)) \
                 + alpha_ * (self.beta - beta_) / beta_
        return kl_div
