#-*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import igraph as ig
import base
import run


if __name__ == "__main__":
    sim = base.AxSimulation('k_plus_a2', 4.0, 3, 4, [])
    base.__a = 1
    sim.set_a(1)
    # g = base.AxGraph.random_graph_with_attrs(N=1000, q=9000)
    # g = sim.basic_algorithm(g, 4000000)

    k = {}
    for i in range(100):
        print i
        g = base.AxGraph.random_graph_with_attrs(N=500, q=21)
        g = sim.basic_algorithm(g, 1000000)
        for x, _, y in g.degree_distribution().bins():
            if x in k:
                k[x] += y
            else:
                k[x] = y
    base.write_object_to_file(k, 'rozklad_k_k222plusa1_500_q21_sr_po100___')

    print k
    plt.scatter(k.keys()[1:], k.values()[1:])
    plt.ylim([1, 15000])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
