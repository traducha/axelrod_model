#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy.stats.mstats import chisquare
from scipy.stats import kstest, ks_2samp
import scipy
import numpy as np
import math
import pprint
import base
mpl.rcParams['font.family'] = 'serif'

q = 2  # 2, 80, 150, 5000
# mode = 'BA'  # 'normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2'
func = 'poisson'  # poisson, norm, expon, pareto
# scale = 'log'  # normal, ylog, log
# skip_first = 0  # how many to skip
# norm_all = False

funcs = ['poisson', 'norm', 'expon', 'pareto']
q_list = [2, 80, 150, 5000]


class Distributions:

    @staticmethod
    def poisson(k, lambda_):
        if isinstance(k, (list, tuple, np.ndarray)):
            return [1.0 * (lambda_**k_) * (np.exp(-lambda_)) / (math.factorial(k_)) for k_ in k]
        return 1.0 * (lambda_**k) * (np.exp(-lambda_)) / (math.factorial(k))

    @staticmethod
    def expon(x, lambda_):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [1.0 * lambda_ * np.exp(-lambda_ * x_) for x_ in x]
        return 1.0 * lambda_ * np.exp(-lambda_ * x)

    @staticmethod
    def norm(x, miu, sigma):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [(1.0 / (np.sqrt(2.0 * math.pi * (sigma**2.0)))) * np.exp(-((x_ - miu)**2.0) / (2 * (sigma**2.0))) for x_ in x]
        return (1.0 / (np.sqrt(2.0 * math.pi * (sigma**2.0)))) * np.exp(-((x - miu)**2.0) / (2 * (sigma**2.0)))

    @staticmethod
    def pareto(x, x_min, alpha):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [(1.0 * alpha * (x_min**alpha)) / (x_**(alpha + 1.0)) for x_ in x]
        return (1.0 * alpha * (x_min**alpha)) / (x**(alpha + 1.0))


def fit_one_plot(func, q, mode, show=True, scale='normal', norm_all=False, skip_first=None):
    if q == 5000:
        path = '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/degree_dist/dynamical'
    else:
        path = '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/degree_dist'

    f_name1 = '{}/{}_degree_N500_q{}_av100.data'.format(path, mode, q)
    f_name2 = '{}/{}_degree_N1000_q{}_av100.data'.format(path, mode, q)
    f_name3 = '{}/{}_degree_N2000_q{}_av100.data'.format(path, mode, q)
    k1 = base.read_object_from_file(f_name1)
    k2 = base.read_object_from_file(f_name2)
    k3 = base.read_object_from_file(f_name3)

    k1_list, k2_list, k3_list, = [], [], []
    for k, list_ in [(k1, k1_list), (k2, k2_list), (k3, k3_list)]:
        for degree, times in k.items()[skip_first:]:
            list_.extend([int(degree) for _ in xrange(int(times*1))])
    # print(k3_list[-10:])

    for k in [k1, k2, k3]:
        skip = 0 if norm_all else skip_first
        norm = sum(k.values()[skip:]) * 1.0
        for key, value in k.items():
            k[key] = value * 1.0 / norm

    plt.scatter(k1.keys()[skip_first:], k1.values()[skip_first:], marker='o', facecolors='none', edgecolors='b', s=20 * 2)
    plt.scatter(k2.keys()[skip_first:], k2.values()[skip_first:], marker='s', facecolors='none', edgecolors='r', s=20 * 2)
    plt.scatter(k3.keys()[skip_first:], k3.values()[skip_first:], marker='^', facecolors='none', edgecolors='g', s=20 * 2)

    res = {
        500: {},
        1000: {},
        2000: {}
    }

    for i, k in enumerate([k1, k2, k3]):
        N = 500 if not i else (1000 if i == 1 else 2000)
        function = getattr(Distributions, func)
        x, y = k.keys()[skip_first:], k.values()[skip_first:]
        try:
            popt, pcov = fit(function, x, y)
            plt.plot(x, function(x, *popt), '-', color='black')
            # print('Parameters for N={} are: {}, with standard deviation: {}'.format(N, popt, np.sqrt(np.diag(pcov))))

            ss_res = np.sum([(y_ - function(x_, *popt)) ** 2.0 for x_, y_ in zip(x, y)])
            y_mean = np.mean(y)
            ss_tot = np.sum([(y_ - y_mean) ** 2.0 for y_ in y])
            r_squared = 1.0 - (ss_res / ss_tot)
            # print('R^2 of the fit is: {}'.format(round(r_squared, 3)))
        # print('Goodness-of-fit Chi^2: {}, and p-value: {}'.format(*chisquare(y, function(x, *popt), ddof=len(popt))))

        # m, a = popt
        # sample = []
        # for s in ((np.random.pareto(a, len([k1_list, k2_list, k3_list][i])) + 1) * m):
        #     if s < m:
        #         continue
        #         sample.append(m)
        #     else:
        #         sample.append(s)

            D, p_value = kstest([k1_list, k2_list, k3_list][i], func, args=popt)
            # print('Goodness-of-fit KS - D and p-value: {}'.format(D, p_value))
            # # print('Goodness-of-fit KS - D and p-value: {}'.format(ks_2samp([k1_list, k2_list, k3_list][i], sample)))
            # print max(sample)
            res[N]['r2'] = round(r_squared, 3)
            # res[N]['D'] = round(D, 3)
            # res[N]['p'] = round(p_value, 3)
        except:
            res[N]['r2'] = None
            # res[N]['D'] = None
            # res[N]['p'] = None

    if scale == 'ylog':
        plt.yscale('log')
    elif scale == 'log':
        plt.yscale('log')
        plt.xscale('log')

    plt.title('mode={}, q={}, fit={}'.format(mode, q, func))
    plt.ylabel(r'$P(k)$')
    plt.xlabel(r'$k$')
    if show:
        plt.show()
    plt.clf()
    return res


funcs = ['poisson', 'expon', 'norm', 'pareto']
q_list = [2, 80, 150, 5000]
show = 1
skip_first = 2  # how many to skip
norm_all = 0
scale = 'normal'  # normal, ylog, log
mode = 'cluster'  # 'normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2'
if __name__ == '__main__':
    for q in [5000]:  # 2, 80, 150, 5000
        for func in funcs:
            res = fit_one_plot(func, q, mode, show=show, scale=scale, norm_all=norm_all, skip_first=skip_first)
            print
            print(func)
            print(q)
            pprint.pprint(res)
            try:
                print(np.mean([res[500]['r2'], res[1000]['r2'], res[2000]['r2']]))
            except:
                print(None)


