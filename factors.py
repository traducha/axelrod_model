#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import logging as log
import numpy as np
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
N_list = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
T = 8000000
processes = 8
q = 3
av_over = 24
modes = ['normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2']


def get_average_path(N, mode, av_over):
    sim = base.AxSimulation(mode, 4.0, 3, processes, [])
    base.__a = 1
    sim.set_a(1)

    graphs = sim.return_many_graphs_multi(N, q, T, av_over)
    path = []
    for g in graphs:
        path.append(g.average_path_length())
    return 1.0 * sum(path) / (len(path) * 1.0)


def average_path_modes_and_n(N_list, modes, av_over):
    for mode in modes:
        paths = {}
        for N in N_list:
            main_time = time.time()
            paths[N] = get_average_path(N, mode, av_over)
            log.info("computation of components for mode %s and q %s finished in %s min" %
                     (mode, q, round((time.time()-main_time)/60.0, 2)))
        base.write_object_to_file(paths, mode + '_paths_q' + str(q) + '_av' + str(av_over) + '.data')
    return

if __name__ == '__main__':
    pass
    # average_path_modes_and_n(N_list, modes, av_over)

r = base.read_object_from_file('normal_clustering_N500_q22_av500.data')
plt.scatter(r.keys(), r.values())
plt.show()