import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd

from vinf.utils import load_stock_growth, downsample


mpl.rc('text', usetex=True)
plt.rcParams['figure.figsize'] = [4.4, 4]
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')


def mse_grad_gamma_plot():
    with open('mse_grad_gamma.pkl', 'rb') as f:
        results = pickle.load(f)
    map = {'VIND-1': 'VIND $\epsilon=1$', 'VIND-5': 'VIND $\epsilon=5$',
           'VIND-10': 'VIND $\epsilon=10$', 'VIND-100': 'VIND $\epsilon=100$'}
    its = results['iterations']
    biases = results['biases']
    variances = results['variances']
    metrics = results['metrics']
    for k in biases:
        bias = pd.Series(biases[k])
        plt.plot(its, bias.rolling(20).mean(), label=map.get(k, k))
    plt.legend()
    plt.yscale('log')
    plt.ylabel('bias squared')
    plt.xlabel('iteration')
    plt.tight_layout()
    plt.savefig('plots/bias_grad_gamma.pdf')
    plt.show()
    for k in biases:
        plt.plot(its, variances[k], label=map.get(k, k))
    plt.legend()
    plt.yscale('log')
    plt.ylabel('variance')
    plt.xlabel('iteration')
    plt.tight_layout()
    plt.savefig('plots/var_grad_gamma.pdf')
    plt.show()
    for k in biases:
        plt.plot(its, np.array(biases[k])+np.array(variances[k]), label=map.get(k, k))
    plt.legend()
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel('iteration')
    plt.tight_layout()
    plt.savefig('plots/mse_grad_gamma.pdf')
    plt.show()
    plt.plot(its, metrics['KL'])
    plt.xlabel('iteration')
    plt.ylabel('KL-Divergence')
    plt.tight_layout()
    plt.savefig('plots/mse_grad_gamma_kl.pdf')
    plt.show()
    plt.plot(its, metrics['df'])
    plt.xlabel('iteration')
    plt.ylabel('df')
    plt.tight_layout()
    plt.savefig('plots/mse_grad_gamma_df.pdf')
    plt.show()


def linear_regression_plot():
    taken = ['BBVI-RP', 'VIND-0_1', 'VIND-0_5-Ent', 'VIND-1', 'VIND-2-Ent']
    map = {'BBVI-RP': 'BBVI-RP', 'VIND-0_1': 'VIND $\epsilon=0.1$',
           'VIND-0_5-Ent': 'VIND $\epsilon=0.5$', 'VIND-1': 'VIND $\epsilon=1$',
           'VIND-2-Ent': 'VIND $\epsilon=2$'}
    with open('linear_regression.pkl', 'rb') as f:
        res = pickle.load(f)
    print(res.keys())
    for k in res:
        if k not in taken:
            continue
        plt.plot(-np.array(res[k]['elbo']), label=map[k])
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('negative ELBO')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/linear_regression_elbo.pdf')
    plt.show()
    for k in res:
        if k not in taken:
            continue
        plt.plot(-np.array(res[k]['testloss']), label=map[k])
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('negative test log loss')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/linear_regression_testloss.pdf')
    plt.show()


def wishart_student_normal_plot():
    n_samples_log_loss = 100
    map = {'BBVI-RP': 'BBVI-RP', 'BBVI-RP-wodf': 'BBVI-RP without $p$',
           'VIND-1-': r'VIND $\epsilon_p=2d$ $\epsilon_\alpha = 1$'}
    with open('wishart_student_normal.pkl', 'rb') as f:
        res = pickle.load(f)
    print(res.keys())
    fig, ax = plt.subplots()
    for k in ['BBVI-RP', 'BBVI-RP-wodf', 'VIND-1-']:
        ax.plot(-np.array(res[k]['elbo']), label=map[k])
    ax.legend()
    ax.set_xlabel('iteration')
    ax.set_ylabel('negative ELBO')
    plt.tight_layout()
    plt.savefig('plots/wishart_student_normal_elbo.pdf')
    plt.show()
    for k in ['BBVI-RP', 'BBVI-RP-wodf', 'VIND-1-']:
        plt.plot(-np.array(res[k]['testloss'])/n_samples_log_loss, label=map[k])
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('negative test log loss')
    plt.tight_layout()
    plt.savefig('plots/wishart_student_normal_test.pdf')
    plt.show()


def wishart_normal_normal_plot():
    n_samples_log_loss = 100
    map = {'BBVI-RP': 'BBVI-RP', 'BBVI-RP-wodf': 'BBVI-RP without $p$',
           'VIND-1': r'VIND $\epsilon_p=2d$',
           'VIND-2': r'VIND $\epsilon_p=4d$'}
    with open('wishart_normal_normal.pkl', 'rb') as f:
        res = pickle.load(f)
    print(res.keys())
    for k in ['BBVI-RP', 'BBVI-RP-wodf', 'VIND-1', 'VIND-2']:
        plt.plot(-np.array(res[k]['elbo']), label=map[k])
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('negative ELBO')
    plt.tight_layout()
    plt.savefig('plots/wishart_normal_normal_elbo.pdf')
    plt.show()
    for k in ['BBVI-RP', 'BBVI-RP-wodf', 'VIND-1', 'VIND-2']:
        plt.plot(-np.array(res[k]['testloss'])/n_samples_log_loss, label=map[k])
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('negative test log loss')
    plt.tight_layout()
    plt.savefig('plots/wishart_normal_normal_test.pdf')
    plt.show()


def stock_model_univariate_analysis():
    frame_log_grow = load_stock_growth(log=True, drop_na=True)
    frame_log_grow = downsample(frame_log_grow, 5)
    frame_log_grow -= frame_log_grow.mean()
    stock_name = 'CVX'
    s, e = -0.1, 0.1
    x_vals = np.linspace(s, e, 1000)
    mu, sigma = sp.stats.norm.fit(frame_log_grow[stock_name].values)
    plt.plot(x_vals, sp.stats.norm.pdf(x_vals, loc=mu, scale=sigma), label='Normal', alpha=.9, lw=2)
    mu, sigma = scipy.stats.laplace.fit(frame_log_grow[stock_name])
    plt.plot(x_vals, sp.stats.laplace.pdf(x_vals, loc=mu, scale=sigma), label='Laplace', alpha=.9, lw=2)
    d, mu, sigma = scipy.stats.t.fit(frame_log_grow[stock_name])
    plt.plot(x_vals, sp.stats.t.pdf(x_vals, d, loc=mu, scale=sigma), label='t-Student', alpha=.9, lw=2)
    plt.hist(frame_log_grow[stock_name], bins=50, density=True, alpha=0.5)
    plt.legend()
    plt.ylabel('hist. frequency')
    plt.xlabel('log return')
    plt.xlim([s, e])
    plt.tight_layout()
    plt.savefig('plots/ml_cvx_stock.pdf')
    plt.show()


if __name__ == '__main__':
    #stock_model_univariate_analysis()
    #mse_grad_gamma_plot()
    #linear_regression_plot()
    wishart_student_normal_plot()
    wishart_normal_normal_plot()
