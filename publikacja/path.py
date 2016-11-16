#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit as fit
import base
mpl.rcParams['font.family'] = 'serif'
print(mpl.rcParams['figure.figsize'])  # default is [8, 6]


ticksize = 14
axsize = 16
modes = ['BA', 'k_plus_a', 'k_plus_a2', 'cluster']
N_list = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]


def ln(x, a):
    if isinstance(x, list):
        return [a * np.log(i) for i in x]
    return a * np.log(x)


def lin(x, a, b):
    if isinstance(x, list):
        return [a * i + b for i in x]
    return a * x + b


def plot_path(mode):
    r = base.read_object_from_file('/home/tomaszraducha/Dropbox/Dane/mgr/mgr/path/{}_paths_q3_av48.data'.format(mode))
    y = []
    y2 = []
    for N in N_list:
        y.append(r[N])
    plt.scatter(N_list, y, marker='s', s=20*2)

    if mode in ['BA', 'k_plus_a']:
        popt, pcov = fit(ln, N_list, y)
        plt.plot(range(410, 4100, 2), ln(range(410, 4100, 2), popt[0]), 'k--')
    elif mode == 'cluster':
        popt, pcov = fit(lin, N_list, y)
        plt.plot(range(410, 4100, 2), lin(range(410, 4100, 2), popt[0], popt[1]), 'k-')

    plt.xlim([300, 4200])
    plt.xlabel(r'$N$', fontsize=14)
    plt.ylabel(r'$\langle l \rangle$', fontsize=14)
    plt.show()
    # plt.savefig('path/' + mode + '_paths_q3_av48.pdf', format='pdf', bbox_inches='tight')
    plt.clf()
# for mode in modes:
#     plot_path(mode)

mode_sublot = {'BA': 221, 'k_plus_a': 222, 'k_plus_a2': 223, 'cluster': 224}
fig = plt.figure()

for i, mode in enumerate(modes):
    ax = fig.add_subplot(mode_sublot[mode])
    r = base.read_object_from_file('/home/tomaszraducha/Dropbox/Dane/mgr/mgr/path/{}_paths_q3_av48.data'.format(mode))
    y = []
    y2 = []
    for N in N_list:
        y.append(r[N])
    ax.scatter(N_list, y, marker='s', s=20 * 2)

    if mode in ['BA', 'k_plus_a']:
        popt, pcov = fit(ln, N_list, y)
        ax.plot(range(410, 4100, 2), ln(range(410, 4100, 2), popt[0]), 'k--')
    elif mode == 'cluster':
        popt, pcov = fit(lin, N_list, y)
        ax.plot(range(410, 4100, 2), lin(range(410, 4100, 2), popt[0], popt[1]), 'k-')

    ymin, ymax = 0, 10
    if mode == 'k_plus_a':
        ymin, ymax = 4.3, 6.2
        ax.set_ylim([4.3, 6.2])
    elif mode == 'cluster':
        ymin, ymax = 7.5, 26
        ax.set_ylim([7.5, 26])
        ax.set_yticks([8, 12, 16, 20, 24])
    elif mode == 'BA':
        ymin, ymax = 4.1, 5.83
        ax.set_ylim([4.1, 5.83])
        ax.set_yticks([4.2, 4.6, 5.0, 5.4, 5.8])
    elif mode == 'k_plus_a2':
        ymin, ymax = 2.9, 3.55
        ax.set_ylim([2.9, 3.55])

    # ax.set_xticks([500, 1500, 2500, 3500])
    ax.set_xticks([1000, 2000, 3000, 4000])
    ax.set_xlim([300, 4300])
    if i in [2, 3]:
        ax.set_xlabel(r'$N$', fontsize=axsize)
    if i in [0, 2]:
        ax.set_ylabel(r'$\langle l \rangle$', fontsize=axsize)

    from_top = 150 * (ymax - ymin) / 1000.0  # first number is a distance from the top
    absolute = ymax - from_top
    if i == 0:
        ax.text(1000, absolute, r'A', fontsize=axsize)
    elif i == 1:
        ax.text(1000, absolute, r'B', fontsize=axsize)
    elif i == 2:
        ax.text(1000, absolute, r'C', fontsize=axsize)
    elif i == 3:
        ax.text(1000, absolute, r'D', fontsize=axsize)

    if i == 2:
        ax.get_yaxis().set_ticks([2.9, 3.1, 3.3, 3.5])
    # zmiana rozmiaru czcionki tiksów !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ##################################################################
    ax.tick_params(axis='both', which='major', labelsize=ticksize)  # standard 12
    ##################################################################
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_y(-0.01)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_x(-0.01)

plt.tight_layout()
# for end in ['pdf', 'svg']:
#     plt.savefig('/home/tomaszraducha/Pulpit/path.{}'.format(end), format=end, bbox_inches='tight')
plt.show()
plt.clf()
