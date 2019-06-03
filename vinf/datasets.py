import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import torch
from vinf.utils import downsample, load_stock_growth


class GammaNormal:
    def __init__(self, mu, tau):
        self.mu = mu
        self.tau = tau

    def generate(self, n, seed=None):
        np.random.seed(seed)
        data = self.mu + np.random.randn(n) * np.sqrt(1/self.tau)
        data = data.astype(np.float64)
        return torch.from_numpy(data)


def BostonHousing(decorrelate, std=True, add_ones=True, dim=20):
    X, y = load_boston(return_X_y=True)
    X, X_test, y, y_test = train_test_split(X, y)
    mean = X.mean(axis=0)
    X, X_test = (X - mean), (X_test - mean)
    if std:
        Xstd = X.std(axis=0)
        X = X / Xstd
        X_test = X_test / Xstd
    if decorrelate:
        U, S, V = np.linalg.svd(X)
        X = X.dot(V.T[:, :dim])
        X_test = X_test.dot(V.T[:, :dim])
    if add_ones:
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return (torch.from_numpy(X.astype(np.float64)), torch.from_numpy(y.reshape(-1, 1).astype(np.float64)),
            torch.from_numpy(X_test.astype(np.float64)), torch.from_numpy(y_test.reshape(-1, 1).astype(np.float64)))


def DowJones(sub_mean=True, dim=20, factor=1e3, offset=110):
    data = load_stock_growth(log=True, drop_na=True)
    data = downsample(data=data, sample_step=5)
    data = data.iloc[offset:]
    X = factor * data.values
    if sub_mean:
        print("WARNING: should not subtract mean on training data.")
        X = X - X.mean(axis=0)
    if dim != 29:
        stock_select = np.random.choice(X.shape[1], dim, replace=False)
        X = X[:, stock_select]
    X, X_test = X[:300], X[300:]
    return torch.from_numpy(X.astype(np.float64)), torch.from_numpy(X_test.astype(np.float64))
