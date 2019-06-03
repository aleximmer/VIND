import os
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam, Adagrad, Adadelta, ASGD, SGD


data_dir = 'data/'


def get_optimizer_dict():
    return {'Adam': Adam, 'Adagrad': Adagrad,
            'Adadelta': Adadelta, 'ASGD': ASGD, 'SGD': SGD}


def get_optimizer(name):
    d = get_optimizer_dict()
    return d[name]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cast(x):
    return torch.from_numpy(x.astype(np.float64))


def tl2lt(a):
    return list(zip(*a))


def downsample(data, sample_step, style='sum'):
    new_df = pd.DataFrame(columns=data.columns)
    for i in np.arange(0, len(data), sample_step):
        time = data.index[i]
        if style == 'mean':
            new_df.loc[time] = data.iloc[i:i + sample_step].mean()
        elif style == 'sum':
            new_df.loc[time] = data.iloc[i:i + sample_step].sum()
        else:
            raise ValueError(str(style) + ' not in {mean, sum}')
    return new_df


def load_stock_growth(log=True, drop_na=True, style='Open'):
    # style defines if using open or 'Close' values
    frames = []
    for f_name in os.listdir(data_dir):
        if '.csv' not in f_name:
            continue
        frame = pd.read_csv(data_dir + f_name, index_col=0)
        col_name = f_name.split('.')[0]
        f = np.log if log else lambda x: x
        frame[col_name] = f(frame[style].rolling(2, 2).apply(lambda x: x[1] / x[0]))
        frames.append(frame[col_name])
    frames = pd.concat(frames, axis=1)
    if drop_na:
        frames = frames.dropna(axis=0, how='any')
    frames.index = pd.to_datetime(frames.index)
    return frames


def symmetrize(variables):
    for name, value in variables.items():
        if name in {'W', 'Sig', 'S'} and value.grad is not None:
            value.grad = 0.5 * value.grad.t() + 0.5 * value.grad
