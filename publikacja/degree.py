#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
import numpy as np
import base
mpl.rcParams['font.family'] = 'serif'
print(mpl.rcParams['figure.figsize'])  # default is [8, 6]
mpl.rcParams['figure.figsize'] = [8.0, 8.0]


def f(x, a, b):
    return a * (x**b)


def e(x):
    if isinstance(x, list):
        return [0.25 * np.exp(-w * 0.25) for w in x]
    return 0.25 * np.exp(-x * 0.25)


plots = [('BA', 80, 321), ('k_plus_a', 5000, 322), ('k_plus_a2', 2, 323),
         ('k_plus_a2', 80, 324), ('k_plus_a2', 150, 325), ('cluster', 80, 326)]
path1 = '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/degree_dist'
path2 = '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/degree_dist/dynamical'
fig = plt.figure()

for i, p in enumerate(plots):
    path = path1
    if i == 1:
        path = path2
    f_name1 = '{}/{}_degree_N500_q{}_av100.data'.format(path, p[0], p[1])
    f_name2 = '{}/{}_degree_N1000_q{}_av100.data'.format(path, p[0], p[1])
    f_name3 = '{}/{}_degree_N2000_q{}_av100.data'.format(path, p[0], p[1])
    k1 = base.read_object_from_file(f_name1)
    k2 = base.read_object_from_file(f_name2)
    k3 = base.read_object_from_file(f_name3)
    for k in [k1, k2, k3]:
        norm = sum(k.values()) * 1.0
        for key, value in k.items():
            k[key] = value * 1.0 / norm

    ax = fig.add_subplot(p[2])
    ax.scatter(k1.keys(), k1.values(), marker='o', facecolors='none', edgecolors='b', s=20 * 2)
    ax.scatter(k2.keys(), k2.values(), marker='s', facecolors='none', edgecolors='r', s=20 * 2)
    ax.scatter(k3.keys(), k3.values(), marker='^', facecolors='none', edgecolors='g', s=20 * 2)
    ymin = 100
    xmin = 100
    xmax = 100
    for k in [k1, k2, k3]:
        ymin = min(ymin, min(k.values(), key=lambda x: x or 10000.0) / 2.0)
        xmin = min(xmin, min(k.keys(), key=lambda x: x or 10000.0) / 1.5)
        xmax = max(xmax, max(k.keys(), key=lambda x: x or 0.01) * 1.5)
    ax.set_ylim(ymin=ymin)

    if i != 5:
        ax.set_yscale('log')
    if p[0] == 'k_plus_a2':
        x, y = k3.keys(), k3.values()
        popt, pcov = fit(f, x[6:], y[6:])
        ax.plot([x[3], x[115]], [f(x[3], popt[0] * 15, popt[1]), f(x[115], popt[0] * 15, popt[1])], '--', color='black')
        ax.text(25, f(x[25], popt[0] * 30, popt[1]), r'$\alpha = {}$'.format(round(-popt[1], 2)), fontsize=14)
        ax.set_xscale('log')
        ax.set_xlim([xmin, xmax])
    elif p[0] == 'BA':
        ax.set_xlim([0, 70])
        ax.plot(k3.keys(), e(k3.keys()), 'k--')
    elif p[0] == 'k_plus_a':
        ax.set_xlim([0, 40])
        ax.plot(k3.keys(), e(k3.keys()), 'k--')
    elif p[0] == 'cluster':
        ax.set_yscale('log')
        ax.set_xlim([0, 32])
        ax.plot(k3.keys(), e(k3.keys()), 'k--')

    if i in [0, 2, 4]:
        ax.set_ylabel(r'$P(k)$', fontsize=14)
    if i > 3:
        ax.set_xlabel(r'$k$', fontsize=14)

plt.tight_layout()
# for end in ['pdf', 'svg']:
#     plt.savefig('/home/tomaszraducha/Pulpit/degree.{}'.format(end), format=end, bbox_inches='tight')
plt.show()
