#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import logging as log
import numpy as np
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
N_list = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]#, 5000, 7000]
T = 10000000
processes = 8
q = 3
av_over = 48
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
            log.info("computation of components for mode %s and N %s finished in %s min" %
                     (mode, N, round((time.time()-main_time)/60.0, 2)))
        base.write_object_to_file(paths, mode + '_paths_q' + str(q) + '_av' + str(av_over) + '.data')
    return

if __name__ == '__main__' and 0:
    main_t = time.time()
    average_path_modes_and_n(N_list, modes, av_over)
    log.info("main function executed in %s min" % round((time.time()-main_t)/60.0, 2))

for mode in modes:
    r = base.read_object_from_file('path/' + mode + '_paths_q3_av48.data')
    y = []
    y2 = []
    for N in N_list:
        y.append(r[N])
        y2.append(np.log(N)/1.3)
    plt.scatter(N_list, y, color='blue')
    # plt.plot(N_list, y2, color='green')
    plt.title('mode ' + mode + ', q=3')
    plt.xlabel('N')
    plt.ylabel('<l>')
    # plt.show()
    plt.savefig('path/' + mode + '_paths_q3_av48.png')
    plt.clf()