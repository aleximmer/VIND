import numpy as np
import torch
from torch.distributions import Gamma, MultivariateNormal, Normal
from vinf.proposals import Wishart, MVtS, InGammaMVN

EPS = 1e-15


class GammaNormal:
    def __init__(self, data, mu, df_prior, rate_prior):
        """
        :param data: Torch Tensor
        :param mu: float
        :param df_prior: float
        :param rate_prior: float
        """
        self.prior = Gamma(torch.tensor([df_prior], dtype=torch.float64),
                           torch.tensor([rate_prior], dtype=torch.float64))
        self.mu = mu
        self.data = data

    def log_prob(self, tau):
        """
        :param tau: Torch Tensor
        :return: log joint pdf
        """
        likelihood = Normal(torch.tensor([self.mu], dtype=torch.float64),
                            torch.sqrt(1 / tau))
        return torch.sum(likelihood.log_prob(self.data)) + self.prior.log_prob(tau)


class WishartGammaNormalStudent:
    def __init__(self, data, mu_prior, alpha_prior, W_df_prior, W_prior, G_df_prior, rate_prior):
        d = W_prior.shape[0]
        self.W_prior = Wishart({'df': torch.tensor([W_df_prior], dtype=torch.float64),
                                'W': torch.from_numpy(W_prior.astype(np.float64))})
        self.nu_prior = Gamma(torch.tensor([G_df_prior], dtype=torch.float64),
                              torch.tensor([rate_prior], dtype=torch.float64))
        self.mu_prior = MultivariateNormal(loc=mu_prior * torch.ones(d, dtype=torch.float64),
                                           covariance_matrix=alpha_prior * torch.eye(d, dtype=torch.float64))
        self.data = data

    def log_prob(self, nuPmu):
        nu, P, mu = nuPmu
        if nu <= EPS:
            nu = torch.tensor([EPS], dtype=torch.float64)
        likelihood = MVtS({'df': nu, 'loc': mu, 'Sig': torch.inverse(P)}, P=P)
        return (torch.sum(likelihood.log_prob(self.data)) + self.W_prior.log_prob(P)
                + self.nu_prior.log_prob(nu) + self.mu_prior.log_prob(mu))

    def llh(self, nuPmu):
        nu, P, mu = nuPmu
        likelihood = MVtS({'df': nu, 'loc': mu, 'Sig': torch.inverse(P)}, P=P)
        return torch.sum(likelihood.log_prob(self.data))


class WishartNormalNormal:
    def __init__(self, data, mu_prior, alpha_prior, W_df_prior, W_prior):
        d = W_prior.shape[0]
        self.W_prior = Wishart({'df': torch.tensor([W_df_prior], dtype=torch.float64),
                                'W': torch.from_numpy(W_prior.astype(np.float64))})
        self.mu_prior = MultivariateNormal(loc=mu_prior * torch.ones(d, dtype=torch.float64),
                                           covariance_matrix=alpha_prior * torch.eye(d, dtype=torch.float64))
        self.data = data

    def log_prob(self, Pmu):
        P, mu = Pmu
        likelihood = MultivariateNormal(mu, precision_matrix=P)
        return (torch.sum(likelihood.log_prob(self.data)) + self.W_prior.log_prob(P)
                + self.mu_prior.log_prob(mu))

    def llh(self, Pmu):
        P, mu = Pmu
        likelihood = MultivariateNormal(mu, precision_matrix=P)
        return torch.sum(likelihood.log_prob(self.data))


class LinearRegression:
    def log_prob(self, thetalam):
        theta, lam = thetalam
        conditional_model = Normal(self.X @ theta, 1 / lam)
        likelihood = torch.sum(conditional_model.log_prob(self.y))
        return likelihood + self.prior.log_prob(thetalam)

    def llh(self, thetalam):
        theta, lam = thetalam
        likelihood = Normal(self.X @ theta, 1 / lam)
        return torch.sum(likelihood.log_prob(self.y))


class MVNGammaLinRegIndep(LinearRegression):
    def __init__(self, data, s_prior_factor, df_prior, rate_prior):
        self.X = data[0]
        self.y = data[1]
        d = self.X.shape[1]
        self.n = len(self.X)
        self.prior = InGammaMVN({
            'loc': torch.zeros(d, dtype=torch.float64),
            'S': s_prior_factor * torch.eye(d, dtype=torch.float64),
            'df': torch.tensor([df_prior], dtype=torch.float64),
            'rate': torch.tensor([rate_prior], dtype=torch.float64)
        })
