#-*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from base import *
mpl.rcParams['font.family'] = 'serif'


type_constants = {
    'BA': {'phase': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/BA/N500',
           'clustering': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/components/BA',
           'first': 19,
           'second': 144},
    'normal': {'phase': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/normal/N500',
               'clustering': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/components/normal',
               'first': 22,
               'second': 389},
    'k_plus_a': {'phase': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/k_plus_a/a1/N500',
                 'clustering': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/components/k_plus_a',
                 'first': 19,
                 'second': 333},
    'k_plus_a2': {'phase': '/home/tomaszraducha/Dropbox/Dane/home2/mgr/k_plus_a2/a1/N500',
                  'clustering': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/components/k_plus_a2',
                  'first': 9,
                  'second': 111},
    'cluster': {'phase': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/cluster/N500',
                'clustering': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/components/cluster',
                'first': 3,
                'second': 129},
}

q_list = [int(1.17 ** i) for i in range(2, 59) if int(1.17 ** i) != int(1.17 ** (i - 1))]  # 71 points in log scale
# l = [94, 111, 129, 152, 177, 208, 243, 284, 333, 389, 456]
# q_list += [l[i]+(l[i+1]-l[i])/3 for i in range(len(l)-1)]\
#             + [l[i]+2*(l[i+1]-l[i])/3 for i in range(len(l)-1)]
# q_list = q_list[::3]
q_list.sort()


def draw(_type):
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

    plt.scatter(Q, d, facecolors='none', edgecolors='r')
    plt.scatter(Q, s, color='blue')
    plt.plot(Q, clust_global, 'g-')
    plt.plot(Q, clust_local, '--', color='#ff5e00')
    plt.plot([type_constants[_type]['first'], type_constants[_type]['first']], [0.0, 1.0], color='black')
    plt.plot([type_constants[_type]['second'], type_constants[_type]['second']], [0.0, 1.0], color='black')
    plt.xlim([1, 10000])
    plt.ylim([0, 1])
    plt.xscale('log')
    # plt.yscale('log')
    plt.show()
    plt.clf()


types = type_constants.keys()
for t in types:
    draw(t)
