#-*- coding: utf-8 -*-

import logging as log
import time
import base

N = 500
T = 3000000
processes = 12
q_list = [2, 80, 150, 5000]
av_over = 100
modes = ['normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2']


def get_degree(q, mode, av_over):
    sim = base.AxSimulation(mode, 4.0, 3, processes, [])
    base.__a = 1
    sim.set_a(1)

    graphs = sim.return_many_graphs_multi(N, q, T, av_over)
    k = {}
    for g in graphs:
        for x, _, y in g.degree_distribution().bins():
            if x in k:
                k[x] += y * 1.0 / (1.0 * av_over)
            else:
                k[x] = y * 1.0 / (1.0 * av_over)
    base.write_object_to_file(k, mode + '_degree_N500_q' + str(q) + '_av' + str(av_over) + '.data')
    return


def degree_for_modes_and_q(q_list, modes, av_over):
    for mode in modes:
        for q in q_list:
            main_time = time.time()
            get_degree(q, mode, av_over)
            log.info("computation of degree for mode %s and q %s finished in %s min" %
                     (mode, q, round((time.time()-main_time)/60.0, 2)))
    return

if __name__ == '__main__':
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    degree_for_modes_and_q(q_list, modes, av_over)

