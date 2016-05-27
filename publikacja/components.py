#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import base
import matplotlib as mpl
from scipy.optimize import curve_fit as fit
mpl.rcParams['font.family'] = 'serif'


modes = ['BA', 'k_plus_a', 'k_plus_a2', 'cluster']

mapping = {
    'normal': {'qc': 22},
    'BA': {'qc': 19},
    'k_plus_a': {'qc': 19},
    'k_plus_a2': {'qc': 10},  # TODO probably we should calculate it for q=9 for this model
    'cluster': {'qc': 3},
}
components_file = '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/components'


def f(x, a, b):
    return a * (x**b)


def plot_comps(dist):
    x = dist.keys()
    y = dist.values()
    popt, pcov = fit(f, x[3:], y[3:])
    print(round(-popt[1], 2))
    plt.scatter(x, y)
    plt.plot([x[3], x[200]], [f(x[3], popt[0]*8, popt[1]), f(x[200], popt[0]*8, popt[1])], '--', color='black')
    plt.text(25, f(x[25], popt[0]*15, popt[1]), r'$\alpha = {}$'.format(round(-popt[1], 2)), fontsize=14)
    plt.xlabel('$S$', fontsize=14)
    plt.ylabel(r'$P(S)$', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin=0.8)
    plt.ylim(ymin=min(dist.values())/2.0)
    plt.ylim(ymax=max(dist.values())*2.0)
    plt.show()
    plt.clf()


# for mode in modes:
#     r = base.read_object_from_file('{}/{}_components_N500_q{}_av3000.data'.format(components_file, mode, mapping[mode]['qc']))
#     plot_comps(r)


mode_sublot = {'BA': 221, 'k_plus_a': 222, 'k_plus_a2': 223, 'cluster': 224}
fig = plt.figure()

for i, mode in enumerate(modes):
    r = base.read_object_from_file('{}/{}_components_N500_q{}_av3000.data'.format(components_file, mode, mapping[mode]['qc']))
    x = r.keys()
    y = r.values()
    popt, pcov = fit(f, x[3:], y[3:])
    ax = fig.add_subplot(mode_sublot[mode])
    ax.scatter(x, y, facecolors='none', edgecolors='b')
    ax.plot([x[3], x[200]], [f(x[3], popt[0] * 8, popt[1]), f(x[200], popt[0] * 8, popt[1])], '--', color='black')
    ax.text(25, f(x[25], popt[0] * 15, popt[1]), r'$\alpha = {}$'.format(round(-popt[1], 2)), fontsize=14)
    if i in [2, 3]:
        ax.set_xlabel('$S$', fontsize=14)
    if i in [0, 2]:
        ax.set_ylabel(r'$P(S)$', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xmin=0.8)
    ax.set_ylim(ymin=min(r.values()) / 2.0)
    ax.set_ylim(ymax=max(r.values()) * 2.0)

plt.tight_layout()
# for end in ['pdf', 'svg']:
#     plt.savefig('/home/tomaszraducha/Pulpit/components.{}'.format(end), format=end, bbox_inches='tight')
plt.show()
