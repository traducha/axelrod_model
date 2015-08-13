#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import logging as log
import numpy as np
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
N = 500
T = 3000000
processes = 8
q_list = [3, 10, 19, 22, 70]
av_over = 2000
modes = ['normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2']


def get_components_dist(q, mode, av_over):
    sim = base.AxSimulation(mode, 4.0, 3, processes, [])
    base.__a = 1
    sim.set_a(1)

    graphs = sim.return_many_graphs_multi(N, q, T, av_over)
    dist = {}
    clustering = {'global': 0.0, 'av_local_nan': 0.0, 'av_local_zero': 0.0}
    for g in graphs:
        clustering['global'] += g.transitivity_undirected() / (1.0 * av_over)
        clustering['av_local_nan'] += g.transitivity_avglocal_undirected(mode="nan") / (1.0 * av_over)
        clustering['av_local_zero'] += g.transitivity_avglocal_undirected(mode="zero") / (1.0 * av_over)
        for comp in g.get_components().values():
            if comp in dist:
                dist[comp] += 1.0 / (1.0 * av_over)
            else:
                dist[comp] = 1.0 / (1.0 * av_over)
    base.write_object_to_file(dist, mode + '_components_N500_q' + str(q) + '_av' + str(av_over) + '.data')
    base.write_object_to_file(clustering, mode + '_clustering_N500_q' + str(q) + '_av' + str(av_over) + '.data')
    return


def components_dis_for_modes_and_q(q_list, modes, av_over):
    for mode in modes:
        for q in q_list:
            main_time = time.time()
            get_components_dist(q, mode, av_over)
            log.info("computation of components for mode %s and q %s finished in %s min" %
                     (mode, q, round((time.time()-main_time)/60.0, 2)))
    return


def plot_comps(dist, mode, q):
    plt.scatter(dist.keys(), dist.values())
    plt.title('mode - ' + mode + ' q = ' + str(q))
    plt.xlabel('S')
    plt.ylabel('P(S)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin=0.8)
    plt.ylim(ymin=min(dist.values())/2.0)
    plt.ylim(ymax=max(dist.values())*2.0)
    plt.show()
    plt.clf()

if __name__ == '__main__':
    components_dis_for_modes_and_q(q_list, modes, av_over)
    # dist = {}
    # r = base.read_object_from_file('normal_clustering_N500_q22_av500.data')
    # for i in range(int(len(r)/2.0)):
    #     dist[r.keys()[i]] = r.values()[i] + r.values()[i+1]
    # plot_comps(r, modes[0], q_list[0])