#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from base import *
mpl.rcParams['font.family'] = 'serif'

ticksize = 14 * 0.75 / 0.5
axsize = 16 * 0.75 / 0.5

type_constants = {
    'normal': {'phase': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/normal/N500',
               'clustering': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/components/normal',
               'first': 22,
               'second': 389},
    'BA': {'phase': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/BA/N500',
           'clustering': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/components/BA',
           'first': 19,
           'second': 144},
    'k_plus_a': {'phase': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/k_plus_a/a1/N500',
                 'clustering': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/components/k_plus_a',
                 'first': 19,
                 'second': 333},
    'k_plus_a2': {'phase': '/home/tomaszraducha/Dropbox/DaneAxelrod/home2/mgr/k_plus_a2/a1/N500',
                  'clustering': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/components/k_plus_a2',
                  'first': 9,
                  'second': 111},
    'cluster': {'phase': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/cluster/N500',
                'clustering': '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/components/cluster',
                'first': 3,
                'second': 129},
}

q_list = [int(1.17 ** i) for i in range(2, 59) if int(1.17 ** i) != int(1.17 ** (i - 1))]  # 71 points in log scale
q_list.sort()
styles = ['D', 'o', 's', '^', 'p']
colors = ['b', 'r', 'g', 'm', 'y']


def draw(content=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for j, _type in enumerate(('normal', 'BA', 'k_plus_a', 'k_plus_a2', 'cluster')):
        labels = ('original', 'model A', 'model B', 'model C', 'model D')
        s, d, Q, clust_local, clust_global = [], [], [], [], []
        for q in q_list:
            try:
                x, y, _, _ = read_object_from_file('{}/q={}.data'.format(type_constants[_type]['phase'], q))
                clust = read_object_from_file('{}/{}_clustering_coef_N500_q{}_av400.data'
                                              .format(type_constants[_type]['clustering'], _type, q))
            except Exception as e:
                print('NIE WCZYTAŁO DLA Q={}, BO {}'.format(q, e))
                continue
            s.append(x)
            d.append(y)
            Q.append(q)
            clust_local.append(clust['av_local_nan'])
            clust_global.append(clust['global'])
        size = 30
        if content == 'd':
            ax.set_ylabel('$D/N$', fontsize=axsize)
            name = ax.scatter(Q, d, marker=styles[j], color=colors[j], s=size)
            ax.plot(Q, d, color=colors[j])
            name.set_label(labels[j])
        elif content == 's':
            ax.set_ylabel('$S/N$', fontsize=axsize)
            name = ax.scatter(Q, s, marker=styles[j], color=colors[j], s=size)
            ax.plot(Q, s, color=colors[j])
            name.set_label(labels[j])
        elif content == 'c_local':
            ax.set_ylabel('$C$ (local)', fontsize=axsize)
            name = ax.scatter(Q, clust_local, marker=styles[j], color=colors[j], s=size)
            ax.plot(Q, clust_local, color=colors[j])
            name.set_label(labels[j])
        elif content == 'c_global':
            ax.set_ylabel('$C$ (global)', fontsize=axsize)
            name = ax.scatter(Q, clust_global, marker=styles[j], color=colors[j], s=size)
            ax.plot(Q, clust_global, color=colors[j])
            name.set_label(labels[j])

    ax.set_xlim([1, 10000])
    ax.set_ylim([0, 1])
    ax.set_xscale('log')
    ax.set_xlabel('$q$', fontsize=axsize)

    if content == 'd':
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=1, fontsize=axsize)

    # plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)  # standard 12
    for tick in ax.xaxis.get_majorticklabels():
        # tick.set_verticalalignment("top")
        tick.set_y(-0.01)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_x(-0.01)
    plt.tight_layout()
    # for end in ['pdf', 'svg']:
    #     plt.savefig('/home/tomaszraducha/Pulpit/all_{}.{}'.format(content, end), format=end, bbox_inches='tight')
    plt.show()
    # fig.clf()
    # plt.clf()

for p in ('c_local', 'c_global', 's', 'd'):
    draw(content=p)
