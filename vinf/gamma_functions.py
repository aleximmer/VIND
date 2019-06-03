import torch
from math import pi, log
from scipy.special import digamma


def mvlgamma(x, p):
    """
    :param x: Torch Tensor 1D
    :param p: int
    :return: Torch Tensor
    """
    g = p * (p - 1) / 4 * log(pi)
    for i in range(1, p+1):
        g += torch.lgamma(x + (1 - i) / 2)
    return g


def mvdigamma(x, p):
    """
    :param x: Torch Tensor 1D
    :param p: int
    :return: Torch Tensor
    """
    g = 0
    for i in range(1, p+1):
        g += torch.digamma(x + (1 - i) / 2)
    return g
