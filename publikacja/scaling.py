#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from base import *
mpl.rcParams['font.family'] = 'serif'
print(mpl.rcParams['figure.figsize'])  # TODO default is [8, 6]


mapping = {
    'cluster': {'phase': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/cluster',
                'first': 3,
                'second': 129},
}
N = [500, 1000, 1500, 2000, 4000]
styles = ['D', 'o', 's', '^', 'p']
colors = ['b', 'r', 'g', 'm', 'y']

q_list = [int(1.17 ** i) for i in range(2, 59) if int(1.17 ** i) != int(1.17 ** (i - 1))]  # 71 points in log scale
# l = [94, 111, 129, 152, 177, 208, 243, 284, 333, 389, 456]
# q_list += [l[i]+(l[i+1]-l[i])/3 for i in range(len(l)-1)]\
#             + [l[i]+2*(l[i+1]-l[i])/3 for i in range(len(l)-1)]
# q_list = q_list[::3]
q_list.sort()


def draw(mode):
    for i, n in enumerate(N):
        s, Q = [], []
        for q in q_list:
            try:
                x, _, _, _ = read_object_from_file('{}/N{}/q={}.data'.format(mapping[mode]['phase'], n, q))
            except Exception as e:
                print('NIE WCZYTAŁO DLA Q={}, BO {}'.format(q, e))
                continue
            s.append(x*(n**0.15))
            Q.append(q/(n**0.511))
        plt.scatter(Q, s, marker=styles[i], color=colors[i], s=20*2)

    plt.xlim([0.01, 1000])
    #plt.ylim([0, 1])
    plt.xscale('log')
    plt.xlabel('$q$', fontsize=14)
    plt.ylabel('$S/N$', fontsize=14)
    # plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()
    # for end in ['pdf', 'svg']:
    #     plt.savefig('/home/tomaszraducha/Pulpit/scaling.{}'.format(end), format=end, bbox_inches='tight')
    plt.show()
    plt.clf()


modes = ['cluster']
for mode in modes:
    draw(mode)
