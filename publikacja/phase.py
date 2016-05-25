#-*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from base import *


TYPE = 'BA'
type_folders = {
    'BA': {'phase': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/BA/N500',
           'clustering': '/home/tomaszraducha/Dropbox/Dane/mgr/mgr/components/BA'},
}


if __name__ == "__main__":
    q_list = [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))]  # 71 points in log scale
    # l = [94, 111, 129, 152, 177, 208, 243, 284, 333, 389, 456]
    # q_list += [l[i]+(l[i+1]-l[i])/3 for i in range(len(l)-1)]\
    #             + [l[i]+2*(l[i+1]-l[i])/3 for i in range(len(l)-1)]
    # q_list = q_list[::3]
    q_list.sort()
    s = []
    d = []
    s_ = []
    d_ = []
    Q = []
    clust_local = []
    clust_global = []
    for q in q_list:
        try:
            x, y, z, w = read_object_from_file('{}/q={}.data'.format(type_folders[TYPE]['phase'], q))
            clust = read_object_from_file('{}/{}_clustering_coef_N500_q{}_av400.data'
                                          .format(type_folders[TYPE]['clustering'], TYPE, q))
        except Exception:
            continue
        s.append(x)
        d.append(y)
        s_.append(z)
        d_.append(w)
        Q.append(q)
        clust_local.append(clust['av_local_nan'])
        clust_global.append(clust['global'])

    plt.scatter(Q, d, color='red')
    plt.scatter(Q, s, color='blue')
    plt.plot(Q, clust_local, 'g-')
    plt.plot(Q, clust_global, 'r--')
    plt.plot([16, 16], [0.0, 1.0], color='black')
    plt.plot([345, 345], [0.0, 1.0], color='black')
    plt.xlim([1, 10000])
    plt.ylim([0, 1])
    plt.xscale('log')
    # plt.yscale('log')
    plt.show()
    plt.clf()
