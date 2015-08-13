#-*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import igraph as ig
import base
import run


q_list = [int(1.17**i) for i in range(2, 59) if int(1.17**i) != int(1.17**(i-1))]
q_list.sort()
c = ['blue', 'red', 'green', 'yellow', 'purple']

if __name__ == "__main__":
    for i, N in enumerate([500, 1000, 1500, 2000]):
        s = []
        d = []
        s_ = []
        d_ = []
        Q = []
        for q in q_list:
            try:
                x, y, z, w = base.read_object_from_file('k_plus_a/N' + str(N) + '/q=%s.data' % q)
            except:
                continue
            s.append(x)
            d.append(y)
            s_.append(z)
            d_.append(w)
            Q.append(q)

        plt.plot(Q, s, color=c[i])
        # plt.scatter(Q, d, color='red')
    plt.xlim([1, 10000])
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.show()
    plt.clf()
