#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import numpy as np
import base
import math
mpl.rcParams['font.family'] = 'serif'
print(mpl.rcParams['figure.figsize'])  # default is [8, 6]
mpl.rcParams['figure.figsize'] = [8.0 / 0.75, 8.0]


ticksize = 14
axsize = 16


def f(x, a, b):
    return a * (x**b)


def e(x):
    if isinstance(x, list):
        return [0.25 * np.exp(-w * 0.25) for w in x]
    return 0.25 * np.exp(-x * 0.25)


class Distribution:

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


plots = [('BA', 80, 331), ('BA', 5000, 332), ('k_plus_a', 2, 333), ('k_plus_a', 5000, 334), ('k_plus_a2', 2, 335),
         ('k_plus_a2', 80, 336), ('k_plus_a2', 150, 337), ('cluster', 80, 338), ('cluster', 5000, 339)]
path1 = '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/degree_dist'
path2 = '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/degree_dist/dynamical'
fig = plt.figure()

for i, p in enumerate(plots):
    path = path1
    if i == 1 or i == 3 or i == 8:
        path = path2
    f_name1 = '{}/{}_degree_N500_q{}_av100.data'.format(path, p[0], p[1])
    f_name2 = '{}/{}_degree_N1000_q{}_av100.data'.format(path, p[0], p[1])
    f_name3 = '{}/{}_degree_N2000_q{}_av100.data'.format(path, p[0], p[1])
    k1 = base.read_object_from_file(f_name1)
    k2 = base.read_object_from_file(f_name2)
    k3 = base.read_object_from_file(f_name3)

    skip_first = 0
    if i == 1:
        skip_first = 1
    elif i == 8:
        skip_first = 2

    for k in [k1, k2, k3]:
        norm = sum(k.values()[skip_first:]) * 1.0
        for key, value in k.items():
            k[key] = value * 1.0 / norm

    ax = fig.add_subplot(p[2])
    ax.scatter(k1.keys()[skip_first:], k1.values()[skip_first:], marker='o', facecolors='none', edgecolors='b', s=20 * 2)
    ax.scatter(k2.keys()[skip_first:], k2.values()[skip_first:], marker='s', facecolors='none', edgecolors='r', s=20 * 2)
    ax.scatter(k3.keys()[skip_first:], k3.values()[skip_first:], marker='^', facecolors='none', edgecolors='g', s=20 * 2)

    ymin = 100
    ymax = 0
    xmin = 100
    xmax = 0
    for k in [k1, k2, k3]:
        ymin = min(ymin, min(k.values()[skip_first:], key=lambda x: x or 10000.0) / 2.0)
        ymax = max(ymax, max(k.values()[skip_first:], key=lambda x: x or 0.01) * 1.5)
        xmin = min(xmin, min(k.keys()[skip_first:], key=lambda x: x or 10000.0) / 1.5)
        xmax = max(xmax, max(k.keys()[skip_first:], key=lambda x: x or 0.01) * 1.5)
    if i not in (1,2,8):
        ax.set_ylim(ymin=ymin)

    if i not in [2]:
        ax.set_yscale('log')
    if p[0] == 'k_plus_a2':
        x, y = k3.keys(), k3.values()
        popt, pcov = fit(f, x[6:], y[6:])
        ax.plot([x[3], x[115]], [f(x[3], popt[0] * 15, popt[1]), f(x[115], popt[0] * 15, popt[1])], '--', color='black')
        ax.text(25, f(x[25], popt[0] * 30, popt[1]), r'$\alpha = {}$'.format(round(-popt[1], 2)), fontsize=axsize)
        ax.set_xscale('log')
        ax.set_xlim([xmin, xmax])
    elif p[0] == 'BA' and p[1] == 80:
        xmin, xmax = 0, 70
        ax.set_xlim([xmin, xmax])
        ax.plot(k3.keys(), e(k3.keys()), 'k-')
    elif p[0] == 'BA' and p[1] == 5000:
        ymin = 0.00001
        ymax = 0.15
        ax.set_ylim([ymin, ymax])
        ax.set_yscale('log')
        xmin, xmax = 0, 85
        ax.set_xlim([xmin, xmax])
        x, y = k3.keys()[skip_first:], k3.values()[skip_first:]
        popt, pcov = fit(Distribution.norm, x, y)
        plt.plot(x, Distribution.norm(x, *popt), '-', color='black')
    elif p[0] == 'k_plus_a' and p[1] == 2:
        ymin = 0.0
        ymax = 0.22
        ax.set_ylim([ymin, ymax])
        xmin, xmax = 0, 26
        ax.set_xlim([xmin, xmax])
        x, y = k3.keys()[skip_first:], k3.values()[skip_first:]
        popt, pcov = fit(Distribution.poisson, x, y)
        plt.plot(range(0,26), Distribution.poisson(range(0,26), *popt), '-', color='black')
    elif p[0] == 'k_plus_a' and p[1] == 5000:
        xmin, xmax = 0, 40
        ax.set_xlim([xmin, xmax])
        ax.plot(k3.keys(), e(k3.keys()), 'k-')
    elif p[0] == 'cluster' and p[1] == 80:
        xmin, xmax = 0, 32
        ax.set_xlim([xmin, xmax])
        ax.plot(k3.keys(), e(k3.keys()), 'k-')
    elif p[0] == 'cluster' and p[1] == 5000:
        ymin = 0.00001
        ymax = 0.15
        ax.set_ylim([ymin, ymax])
        xmin, xmax = 0, 50
        ax.set_xlim([xmin, xmax])
        x, y = k3.keys()[skip_first:], k3.values()[skip_first:]
        popt, pcov = fit(Distribution.norm, x, y)
        plt.plot(x, Distribution.norm(x, *popt), '-', color='black')

    if i in [0, 3, 6]:
        ax.set_ylabel(r'$P(k)$', fontsize=axsize)
    if i > 5:
        ax.set_xlabel(r'$k$', fontsize=axsize)

    from_top = 930 * np.log10(ymax / (ymin or 0.0001)) / 1000.0  # first number is a distance from the top
    absolute = 10 ** (np.log10(ymax) - from_top)
    if p[0] == 'k_plus_a2':
        from_left_log = 40 * np.log10(xmax / xmin) / 1000.0
        x_absolute_log = 10 ** (np.log10(xmin) + from_left_log)
    else:
        from_left = 40 * (xmax - xmin) / 1000.0
        x_absolute = xmin + from_left
    if i == 0:
        ax.text(x_absolute, absolute, r'A, $q=80$', fontsize=axsize)
    elif i == 1:
        ax.text(x_absolute, absolute, r'A, $q=5000$', fontsize=axsize)
    elif i == 2:
        from_top = 150 * (ymax - ymin) / 1000.0  # first number is a distance from the top
        absolute = ymax - from_top
        from_left = 670 * (xmax - xmin) / 1000.0
        x_absolute = xmin + from_left
        ax.text(x_absolute, absolute, r'B, $q=2$', fontsize=axsize)
    elif i == 3:
        ax.text(x_absolute, absolute, r'B, $q=5000$', fontsize=axsize)
    elif i == 4:
        ax.text(x_absolute_log, absolute, r'C, $q=2$', fontsize=axsize)
    elif i == 5:
        ax.text(x_absolute_log, absolute, r'C, $q=80$', fontsize=axsize)
    elif i == 6:
        ax.text(x_absolute_log, absolute, r'C, $q=150$', fontsize=axsize)
    elif i == 7:
        ax.text(x_absolute, absolute, r'D, $q=80$', fontsize=axsize)
    elif i == 8:
        ax.text(x_absolute, absolute, r'D, $q=5000$', fontsize=axsize)

    if i == 0:
        ax.get_xaxis().set_ticks([0, 20, 40, 60])
    elif i == 3:
        ax.get_xaxis().set_ticks([0, 10, 20, 30, 40])
    elif i == 7:
        ax.get_xaxis().set_ticks([0, 10, 20, 30])
    elif i == 8:
        # ax.get_yaxis().set_ticks([0.0001, 0.001, 0.01])
        ax.get_yaxis().set_ticks([0.00001, 0.001, 0.1])
    elif i == 1:
        ax.get_xaxis().set_ticks([0, 20, 40, 60, 80])
        ax.get_yaxis().set_ticks([0.00001, 0.001, 0.1])
        # ax.get_yaxis().set_ticks([0.0001, 0.001, 0.01])
    elif i == 2:
        ax.get_yaxis().set_ticks([0.0, 0.1, 0.2])
    if i not in (1, 2, 8):
        ax.get_yaxis().set_ticks([0.00001, 0.001, 0.1])
    ax.tick_params(axis='both', which='major', labelsize=ticksize)  # standard 12
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_y(-0.01)


plt.tight_layout()
# for end in ['pdf', 'svg']:
#     plt.savefig('/home/tomaszraducha/Pulpit/degree_big.{}'.format(end), format=end, bbox_inches='tight')
plt.show()

