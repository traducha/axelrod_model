#-*- coding: utf-8 -*-

import matplotlib
matplotlib.rcParams.update({'font.family': 'times new roman'})
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
modes = ['normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2']  # 'normal'
N = 500
av_over = 500
q_list = [i*3 for i in range(51)]
q_list[0] = 2
q_list.sort()
T = 3000000
term = 500000
processes = 8
rest = [10.0, 0.2]
log.info("algorithm computing times has started for N=%s, modes=%s" % (N, modes))

if __name__ == '__main__' and 0:
    for mode in modes:
        main_time = time.time()
        sim = base.AxSimulation(mode, 4.0, 3, processes, rest)
        res = sim.get_times_for_qsd(N, T, av_over, q_list, term)
        base.write_object_to_file(res, mode + '_times_N='+str(N)+'_av='+str(av_over)+'.data')
        log.info("computation of times for mode %s and N %s finished in %s min" %
                 (mode, N, round((time.time()-main_time)/60.0, 2)))


r = {}
res = base.read_object_from_file('times/k_plus_a_times_N=500_av=500.data')
for key, value in res.items():
    if key != 'q':
        r[key] = list(np.array(value) / 500.0)
    else:
        r[key] = value


limit = r['q'].index(150) + 1

# main plot for times
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(r['q'][:limit], r['s'][:limit], color='blue')
ax.scatter(r['q'][:limit], r['d'][:limit], color='red')
ax.plot(r['q'][:limit], r['a'][:limit], color='green')
ax.set_xlim([0, r['q'][limit-1]+3])
ax.set_ylim(ymin=0)
ax.set_xlabel('q')
ax.set_ylabel('Convergence time')
ax.set_title('Convergence time (in MC steps) in cluster N=500, av. over 500; std in subplot')

# inside plot of standard errors

rect = [0.35, 0.6, 0.35, 0.35]
box = ax.get_position()
width = box.width
height = box.height
inax_position = ax.transAxes.transform(rect[0:2])
transFigure = fig.transFigure.inverted()
infig_position = transFigure.transform(inax_position)
x = infig_position[0]
y = infig_position[1]
width *= rect[2]
height *= rect[3]
subax = fig.add_axes([x, y, width, height], axisbg='w')
x_labelsize = subax.get_xticklabels()[0].get_size()
y_labelsize = subax.get_yticklabels()[0].get_size()
x_labelsize *= rect[2]**0.5
y_labelsize *= rect[3]**0.5
subax.xaxis.set_tick_params(labelsize=x_labelsize)
subax.yaxis.set_tick_params(labelsize=y_labelsize)

subax.scatter(r['q'][:limit], r['s_std'][:limit], color='blue')
subax.scatter(r['q'][:limit], r['d_std'][:limit], color='red')
subax.plot(r['q'][:limit], r['a_std'][:limit], color='green')
subax.set_xlim([0, r['q'][limit-1]+3])

plt.show()