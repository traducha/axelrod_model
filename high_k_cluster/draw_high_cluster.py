#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from base import *
mpl.rcParams['font.family'] = 'serif'

ticksize = 14 * 0.75 / 0.5
axsize = 16 * 0.75 / 0.5

q_list = [int(1.17 ** i) for i in range(2, 59) if int(1.17 ** i) != int(1.17 ** (i - 1))]  # 71 points in log scale
q_list.sort()
print q_list


def draw(N):
    directory = '/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/high_k_cluster/N{}'.format(N)

    s, d, Q, clust_local, clust_global = [], [], [], [], []
    s_std, d_std = [], []
    for q in q_list:
        try:
            x, y, x_std, y_std = read_object_from_file('{}/q={}.data'.format(directory, q))
            clust = read_object_from_file('{}/{}_clustering_N{}_q{}_av400.data'
                                          .format(directory, 'high_k_cluster', N, q))
        except Exception as e:
            print('NIE WCZYTAŁO DLA Q={}, BO {}'.format(q, e))
            continue
        s.append(x)
        d.append(y)
        s_std.append(x_std*6)
        d_std.append(y_std*6)
        Q.append(q)
        clust_local.append(clust['av_local_nan'])
        clust_global.append(clust['global'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Q, d, facecolors='none', edgecolors='r')
    ax.scatter(Q, s, color='blue')
    # ax.scatter(Q, s_std, color='black')
    ax.plot(Q, clust_local, 'g--')
    ax.plot(Q, clust_global, '-', color='#ff5e00')
    ax.plot([9, 9], [0.0, 1.0], color='black')
    ax.plot([81, 81], [0.0, 1.0], color='black')
    ax.set_xlim([1, 10000])
    ax.set_ylim([0, 1])
    ax.set_xscale('log')
    ax.set_xlabel('$q$', fontsize=axsize)
    ax.set_ylabel('$S/N, D/N, C$', fontsize=axsize)
    # plt.subplots_adjust(wspace=0.0, hspace=0.0)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)  # standard 12
    for tick in ax.xaxis.get_majorticklabels():
        # tick.set_verticalalignment("top")
        tick.set_y(-0.01)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_x(-0.01)

    plt.tight_layout()
    # for end in ['pdf', 'eps']:
    #     plt.savefig('/home/tomaszraducha/Pulpit/high_k_cluster_PHASE_N{}.{}'.format(N, end), format=end, bbox_inches='tight')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    for N in [500]: #, 1000, 2000, 4000]:
        draw(N)
