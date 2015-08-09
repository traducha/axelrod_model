#!/usr/bin/python
#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import logging as log
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
mode = 'normal'
N = 500
av_over = 200
q_list = [i*3 for i in range(51)]
q_list[0] = 2
q_list.sort()
T = 2000000
processes = 6
rest = [10.0, 0.1]
log.info("started algorithm computing number of components and domains for N=%s, mode=%s" % (N, mode))

points = np.logspace(0, 6, num=200) * 5.0
points = [int(p) for p in points]
points = list(set(points))
points.sort()

if __name__ == '__main__' and 0:
    main_time = time.time()
    comps = []
    doms = []
    times = []
    sim = base.AxSimulation(mode, 4.0, 3, processes, rest)
    gs = []
    for i in range(av_over):
        gs.append(base.AxGraph.random_graph_with_attrs(N=500, q=20))

    for i in range(len(points)):
        iterations = points[i] - points[i-1] if i else points[i]
        for j in range(av_over):
            gs[j] = sim.basic_algorithm(gs[j], iterations)
        comps.append((sum([g.get_number_of_components() for g in gs]) / float(av_over)) / 500.0)
        doms.append((sum([g.get_number_of_domains_properly() for g in gs]) / float(av_over)) / 500.0)
        print i, doms[-1]
        times.append(iterations + times[-1] if len(times) else iterations)

    res = {'t': times, 's': comps, 'd': doms}
    base.write_object_to_file(res, 'numbers.data')
    plt.plot(times, comps)
    plt.plot(times, doms)
    plt.xscale('log')
    plt.show()

    log.info("computation of numbers for mode %s and N %s finished in %s min" %
             (mode, N, round((time.time()-main_time)/60.0, 2)))

r = base.read_object_from_file('normal_numbers.data')
plt.plot(r['t'], r['s'])
plt.plot(r['t'], r['d'])
plt.xscale('log')
plt.xlim(xmin=4)
plt.ylim([-0.1, 1.1])
plt.show()