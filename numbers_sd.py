#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import logging as log
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
modes = ['normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2']
N = 500
av_over = 300
q_list = [20, 60, 65, 70, 100, 150]
T = 3000000
processes = 6
rest = [10.0, 0.1]
log.info("started algorithm computing number of components and domains for N=%s, modes=%s" % (N, modes))

points = np.logspace(0, 6, num=200) * 5.0
points = [int(p) for p in points]
points = list(set(points))
points.sort()


def for_one_q(q, mode):
    main_t = time.time()
    comps = []
    doms = []
    times = []
    sim = base.AxSimulation(mode, 4.0, 3, processes, rest)
    gs = []
    for i in range(av_over):
        gs.append(base.AxGraph.random_graph_with_attrs(N=500, q=q))

    for i in range(len(points)):
        iterations = points[i] - points[i-1] if i else points[i]
        for j in range(av_over):
            gs[j] = sim.basic_algorithm(gs[j], iterations)
        comps.append((sum([g.get_number_of_components() for g in gs]) / float(av_over)) / 500.0)
        doms.append((sum([g.get_number_of_domains_properly() for g in gs]) / float(av_over)) / 500.0)
        times.append(iterations + times[-1] if len(times) else iterations)

    res = {'t': times, 's': comps, 'd': doms}
    base.write_object_to_file(res, mode + '_q' + str(q) + '_N500_numbers.data')
    log.info("computation of numbers for mode %s and q = %s finished in %s min" %
             (mode, q, round((time.time()-main_t)/60.0, 2)))


def over_modes_q(modes, q_list):
    for mode in modes:
        for q in q_list:
            for_one_q(q, mode)


if __name__ == '__main__' and 0:
    main_time = time.time()
    over_modes_q(modes, q_list)
    log.info("computation of numbers for modes %s and N %s finished in %s min" %
             (modes, N, round((time.time()-main_time)/60.0, 2)))


# for mode in modes:
#     for q in q_list:
#         r = base.read_object_from_file('numbers_sd/' + mode + '_q' + str(q) + '_N500_numbers.data')
#         plt.plot(r['t'], r['s'])
#         plt.plot(r['t'], r['d'])
#         plt.xscale('log')
#         plt.xlim(xmin=4)
#         plt.ylim([-0.1, 1.1])
#         plt.show()
#         plt.clf()


for mode in modes:
    f, ax = plt.subplots(6, sharex=True, sharey=True, figsize=(8, 10))
    ax[0].set_title('Number of components (blue) and domains (green) for mode ' + mode)
    for i, q in enumerate(q_list):
        r = base.read_object_from_file('/home/tomaszraducha/Dropbox/DaneAxelrod/mgr/mgr/numbers_sd/' + mode + '_q' + str(q) + '_N500_numbers.data')
        ax[i].plot(r['t'], r['s'], color='blue')
        ax[i].plot(r['t'], r['d'], color='green')
        ax[i].set_xscale('log')
        ax[i].set_xlim(xmin=4)
        ax[i].set_ylim([-0.1, 1.1])
        ax[i].annotate('q = ' + str(q), (r['t'][0], r['d'][0]),
                       xytext=(20.0, 0.4)) #, textcoords='axes fraction',)
                       # arrowprops=dict(facecolor='black', shrink=0.05),
                       # fontsize=16,
                       # horizontalalignment='right', verticalalignment='top')
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    # plt.savefig('numbers_sd/' + mode + '_numbers.png')
    plt.show()
    # plt.clf()

