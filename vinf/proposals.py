from torch.distributions import Gamma as Gamma_torch, StudentT as StudentT_torch, MultivariateNormal as MVN_torch
import torch
from math import log, pi
import numpy as np
import scipy as sp
import scipy.stats
from vinf.gamma_functions import mvlgamma, mvdigamma
from vinf.utils import cast, tl2lt


class Gamma(Gamma_torch):
    nu_min = 1e-9

    def __init__(self, variables):
        super().__init__(variables['df'], variables['rate'])

    def sample_pos_neg(self, sample_shape=(1,), eps=1):
        n = sample_shape[0]
        df = self.concentration.detach().numpy()[0]
        scale = 1 / self.rate.detach().numpy()[0]
        if df - eps / 2 < self.nu_min:
            eps = 2 * (df - self.nu_min)
            raise RuntimeWarning('Eps too high leading to small divisors')
        samples_neg = np.random.gamma(df - eps/2, scale, (n, 1))
        samples = samples_neg + np.random.gamma(eps/2, scale, (n, 1))
        samples_pos = samples + np.random.gamma(eps/2, scale, (n, 1))
        return (torch.from_numpy(samples_neg.astype(np.float64)),
                torch.from_numpy(samples.astype(np.float64)),
                torch.from_numpy(samples_pos.astype(np.float64)))


class InGammaMVN:
    """Independent Gamma-MVN"""
    nu_min = 1e-9

    def __init__(self, variables):
        self.mu = variables['loc']
        if 'S' in variables:
            self.S = variables['S']
        elif 'D' in variables:
            self.S = torch.diag(variables['D'])
        self.df = variables['df']
        self.rate = variables['rate']
        self.d = self.mu.shape[0]
        self.mvn = MVN(variables)
        self.gamma = Gamma(variables)

    def sample(self, sample_shape=(1,)):
        n = sample_shape[0]
        df = self.df.detach().numpy()[0]
        mu = self.mu.detach().numpy().reshape(1, self.d)
        S = self.S.detach().numpy()
        rate = self.rate.detach().numpy()[0]
        lam = np.random.gamma(df, 1/rate, (n, 1))
        theta = np.random.randn(n, self.d)
        X = mu + theta.dot(S)
        return tl2lt((torch.from_numpy(X.astype(np.float64)),
                      torch.from_numpy(lam.astype(np.float64))))

    def sample_pos_neg(self, sample_shape=(1,), eps=1):
        n = sample_shape[0]
        df = self.df.detach().numpy()[0]
        mu = self.mu.detach().numpy().reshape(1, self.d)
        S = self.S.detach().numpy()
        rate = self.rate.detach().numpy()[0]
        lam_neg = np.random.gamma(df - eps/2, 1 / rate, (n, 1))
        lam = lam_neg + np.random.gamma(eps/2, 1 / rate, (n, 1))
        lam_pos = lam + np.random.gamma(eps/2, 1 / rate, (n, 1))
        theta = np.random.randn(n, self.d)
        scaled_theta = theta.dot(S)
        X = mu + scaled_theta
        return (tl2lt((cast(X), cast(lam_neg))),
                tl2lt((cast(X), cast(lam))),
                tl2lt((cast(X), cast(lam_pos))))

    def log_prob(self, thetalam):
        theta, lam = thetalam
        return self.mvn.log_prob(theta) + self.gamma.log_prob(lam)

    def entropy(self):
        return self.mvn.entropy() + self.gamma.entropy()


class MVtS:
    # Multivariate t Student with single shared degree of freedom 'df'
    nu_min = 1e-15

    def __init__(self, variables, P=None):
        self.df = variables['df']
        assert self.df.detach().item() >= self.nu_min
        self.loc = variables['loc']
        self.d = self.loc.shape[0]
        if 'Sig' in variables:
            self.S = variables['Sig']
            self.sig = True
        elif 'Dig' in variables:
            self.S = torch.diag(variables['Dig'])
            self.sig = True
        elif 'S' in variables:
            self.S = variables['S']
            self.sig = False
        elif 'D' in variables:
            self.S = torch.diag(variables['D'])
            self.sig = False
        else:
            raise ValueError('No covariance')
        if P is None:
            self.P = torch.inverse(self.S) if self.sig else torch.inverse(self.S @ self.S)
        else:
            self.P = P
        self.coeff = 0.5 if self.sig else 1.

    def sample(self, sample_shape=(1,)):
        n = sample_shape[0]
        df = self.df.detach().numpy()[0]
        mu = self.loc.detach().numpy().reshape(1, self.d)
        S = self.S.detach().numpy()
        if self.sig:
            S = sp.linalg.sqrtm(S)
        samples = np.random.randn(n, self.d)
        chis = np.random.chisquare(df, (n, 1))
        samples = samples / np.sqrt(chis / df)
        return torch.from_numpy((mu + samples.dot(S)).astype(np.float64))

    def sample_pos_neg(self, sample_shape=(1,), eps=1):
        n = sample_shape[0]
        df = self.df.detach().numpy()[0]
        mu = self.loc.detach().numpy().reshape(1, self.d)
        S = self.S.detach().numpy()
        if self.sig:
            S = sp.linalg.sqrtm(S)
        norm_samples = np.random.randn(n, self.d).dot(S)
        chis_neg = np.random.chisquare(df - eps/2, (n, 1))
        chis_eps_1 = np.random.chisquare(eps / 2, (n, 1))
        chis_eps_2 = np.random.chisquare(eps / 2, (n, 1))
        chis = chis_neg + chis_eps_1
        chis_pos = chis + chis_eps_2
        samples_neg = mu + norm_samples / np.sqrt(chis_neg / (df - eps / 2))
        samples = mu + norm_samples / np.sqrt(chis / df)
        samples_pos = mu + norm_samples / np.sqrt(chis_pos / (df + eps / 2))
        return (torch.from_numpy(samples_neg.astype(np.float64)),
                torch.from_numpy(samples.astype(np.float64)),
                torch.from_numpy(samples_pos.astype(np.float64)))

    def log_prob(self, theta):
        diff = theta - self.loc
        if len(diff.shape) > 1:
            # print('loc so small it cannot be removed, maybe just center
            # if its not already done and leave loc to 0')
            y = torch.stack([delta @ self.P @ delta for delta in diff])
        else:
            y = diff @ self.P @ diff
        lp = -0.5 * (self.df + self.d) * torch.log1p(y / self.df)
        Z = (self.coeff * torch.logdet(self.S) +
             0.5 * self.d * log(pi) +
             0.5 * self.d * torch.log(self.df) +
             torch.lgamma(0.5 * self.df) -
             torch.lgamma(0.5 * (self.df + self.d)))
        return lp - Z

    def entropy(self):
        simple_tst = StudentT_torch(self.df)
        H = self.coeff * torch.logdet(self.S) + self.d * simple_tst.entropy()
        return H


class MVN(MVN_torch):

    def __init__(self, variables):
        if 'Sig' in variables:
            S = variables['Sig']
        elif 'Dig' in variables:
            S = torch.diag(variables['Dig'])
        elif 'S' in variables:
            S = variables['S'] @ variables['S']
        elif 'D' in variables:
            S = torch.diag(variables['D']**2)
        else:
            raise ValueError('No covariance')
        super().__init__(variables['loc'], S)

    def entropy(self):
        H = super().entropy()
        return H.reshape(1)


class Wishart:

    def __init__(self, variables):
        self.p = int(variables['W'].shape[-1])
        self.logdetW = torch.logdet(variables['W'])
        self.C = -variables['df'] * 0.5 * (self.logdetW + self.p * log(2)) - mvlgamma(variables['df'] / 2, self.p)
        self.W = variables['W']
        self.W_inv = torch.inverse(self.W)
        self.df = variables['df']

    def sample(self, sample_shape=(1,)):
        samples = sp.stats.wishart.rvs(self.df.detach().numpy()[0], self.W.detach().numpy(),
                                       size=sample_shape[0]).astype(np.float64)
        if sample_shape[0] == 1:
            samples = samples[np.newaxis, :, :].astype(np.float64)
        return torch.from_numpy(samples)

    def sample_pos_neg(self, sample_shape=(1,), eps=1):
        df = self.df.detach().numpy()[0]
        W = self.W.detach().numpy()
        n = sample_shape[0]
        samples_neg = sp.stats.wishart.rvs(df - eps/2, W, size=n)
        W_eps_1 = sp.stats.wishart.rvs(eps/2, W, size=n)
        W_eps_2 = sp.stats.wishart.rvs(eps/2, W, size=n)
        samples = samples_neg + W_eps_1
        samples_pos = samples + W_eps_2
        if sample_shape[0] == 1:
            samples = samples[np.newaxis, :, :]
            samples_pos = samples_pos[np.newaxis, :, :]
            samples_neg = samples_neg[np.newaxis, :, :]
        return (torch.from_numpy(samples_neg.astype(np.float64)),
                torch.from_numpy(samples.astype(np.float64)),
                torch.from_numpy(samples_pos.astype(np.float64)))

    def log_prob(self, X):
        logdetX = torch.logdet(X)
        expon = torch.trace(torch.matmul(self.W_inv, X))
        return self.C + 0.5 * (self.df - self.p - 1) * logdetX - 0.5 * expon

    def entropy(self):
        H = (self.p + 1) / 2 * self.logdetW + 0.5 * self.p * (self.p + 1) * log(2) + mvlgamma(self.df / 2, self.p) \
            - (self.df - self.p - 1) / 2 * mvdigamma(self.df / 2, self.p) + self.df * self.p / 2
        return H


class SqrtWishart(Wishart):

    def __init__(self, variables):
        new_vars = dict()
        new_vars['W'] = variables['W'] @ variables['W'].t()
        new_vars['df'] = variables['df']
        super().__init__(new_vars)


class WishartGamma:

    def __init__(self, variables):
        self.gamma = Gamma({'df': variables['G_df'], 'rate': variables['rate']})
        self.wishart = SqrtWishart({'df': variables['W_df'], 'W': variables['W']})

    def sample(self, sample_shape=(1,)):
        nu_samples = self.gamma.sample(sample_shape)
        P_samples = self.wishart.sample(sample_shape)
        return tl2lt((nu_samples, P_samples))

    def sample_mult_eps(self, sample_shape=(1,), eps=None):
        if eps is None:
            eps = {'G_df': 1, 'W_df': 2 * self.wishart.W.shape[0]}
        nu_neg, nu, nu_pos = self.gamma.sample_pos_neg(sample_shape, eps['G_df'])
        P_neg, P, P_pos = self.wishart.sample_pos_neg(sample_shape, eps['W_df'])
        samples_neg = {'W_df': tl2lt((nu, P_neg)), 'G_df': tl2lt((nu_neg, P))}
        samples_pos = {'W_df': tl2lt((nu, P_pos)), 'G_df': tl2lt((nu_pos, P))}
        samples = tl2lt((nu, P))
        return samples_neg, samples, samples_pos

    def log_prob(self, nuP, blackwell=False):
        nu, P = nuP
        if blackwell:
            return self.wishart.log_prob(P)
        return self.gamma.log_prob(nu) + self.wishart.log_prob(P)

    def entropy(self):
        return self.gamma.entropy() + self.wishart.entropy()


class WishartGammaNormal(WishartGamma):
    def __init__(self, variables):
        d = variables['loc'].shape[0]
        super().__init__(variables)
        self.normal = MVN_torch(loc=variables['loc'],
                                covariance_matrix=variables['alpha'] * torch.eye(d, dtype=torch.double))

    def sample(self, sample_shape=(1,)):
        nu_samples = self.gamma.sample(sample_shape)
        P_samples = self.wishart.sample(sample_shape)
        mu_samples = self.normal.sample(sample_shape)
        return tl2lt((nu_samples, P_samples, mu_samples))

    def sample_mult_eps(self, sample_shape=(1,), eps=None):
        if eps is None:
            eps = {'G_df': 1, 'W_df': 2 * self.wishart.W.shape[0]}
        nu_neg, nu, nu_pos = self.gamma.sample_pos_neg(sample_shape, eps['G_df'])
        P_neg, P, P_pos = self.wishart.sample_pos_neg(sample_shape, eps['W_df'])
        mus = self.normal.sample(sample_shape)
        samples_neg = {'W_df': tl2lt((nu, P_neg, mus)), 'G_df': tl2lt((nu_neg, P, mus))}
        samples_pos = {'W_df': tl2lt((nu, P_pos, mus)), 'G_df': tl2lt((nu_pos, P, mus))}
        samples = tl2lt((nu, P, mus))
        return samples_neg, samples, samples_pos

    def log_prob(self, nuPmu):
        nu, P, mu = nuPmu
        return self.gamma.log_prob(nu) + self.wishart.log_prob(P) + self.normal.log_prob(mu)

    def entropy(self):
        return self.gamma.entropy() + self.wishart.entropy() + self.normal.entropy()


class WishartNormal:
    def __init__(self, variables):
        d = variables['loc'].shape[0]
        self.wishart = SqrtWishart({'df': variables['df'], 'W': variables['W']})
        self.normal = MVN_torch(loc=variables['loc'],
                                covariance_matrix=variables['alpha'] * torch.eye(d, dtype=torch.double))

    def sample(self, sample_shape=(1,)):
        P_samples = self.wishart.sample(sample_shape)
        mu_samples = self.normal.sample(sample_shape)
        return tl2lt((P_samples, mu_samples))

    def sample_pos_neg(self, sample_shape=(1,), eps=None):
        if eps is None:
            eps = {'W_df': 2 * self.wishart.W.shape[0]}
        P_neg, P, P_pos = self.wishart.sample_pos_neg(sample_shape, float(eps))
        mus = self.normal.sample(sample_shape)
        samples_neg = tl2lt((P_neg, mus))
        samples = tl2lt((P, mus))
        samples_pos = tl2lt((P_pos, mus))
        return samples_neg, samples, samples_pos

    def log_prob(self, Pmu):
        P, mu = Pmu
        return self.wishart.log_prob(P) + self.normal.log_prob(mu)

    def entropy(self):
        return self.wishart.entropy() + self.normal.entropy()
