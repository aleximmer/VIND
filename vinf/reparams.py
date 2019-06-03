import torch


class GammaNormalReparam:

    @staticmethod
    def sample_reparam(sample, vars):
        return sample / vars['rate']

    @staticmethod
    def normalize_sample(sample, det_vars):
        return sample * det_vars['rate']


class WishartStudentNormalReparam:

    @staticmethod
    def sample_reparam(sample, vars):
        return (sample[0] / vars['rate'],
                torch.matmul(torch.matmul(vars['W'], sample[1]), vars['W'].t()),
                vars['alpha'] * sample[2] + vars['loc'])

    @staticmethod
    def normalize_sample(sample, det_vars):
        W_inv = torch.inverse(det_vars['W'])
        return (sample[0] * det_vars['rate'],
                torch.matmul(torch.matmul(W_inv, sample[1]), W_inv.t()),
                (sample[2] - det_vars['loc']) / det_vars['alpha'])


class WishartNormalNormalReparam:

    @staticmethod
    def sample_reparam(sample, vars):
        return (torch.matmul(torch.matmul(vars['W'], sample[0]), vars['W'].t()),
                vars['alpha'] * sample[1] + vars['loc'])

    @staticmethod
    def normalize_sample(sample, det_vars):
        W_inv = torch.inverse(det_vars['W'])
        return (torch.matmul(torch.matmul(W_inv, sample[0]), W_inv.t()),
                (sample[1] - det_vars['loc']) / det_vars['alpha'])


class LinRegIndReparam:

    @staticmethod
    def sample_reparam(sample, vars):
        if 'S' in vars:
            S = vars['S']
        else:
            S = torch.diag(vars['D'])
        return (S @ sample[0] + vars['loc'],
                sample[1] / vars['rate'])

    @staticmethod
    def normalize_sample(sample, det_vars):
        if 'S' in det_vars:
            S = det_vars['S']
        else:
            S = torch.diag(det_vars['D'])
        S_inv = torch.inverse(S)
        return (S_inv @ (sample[0] - det_vars['loc']),
                sample[1] * det_vars['rate'])
