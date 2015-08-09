#-*- coding: utf-8 -*-

import logging as log
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
modes = 'normal'  # ['BA', 'cluster', 'k_plus_a', 'k_plus_a2']  # 'normal'
N = 500
av_over = 100
q_list = [i*3 for i in range(51)]
q_list[0] = 2
q_list.sort()
T = 3000000
term = 500000
processes = 8
rest = [10.0, 0.1]
log.info("algorithm computing times has started for N=%s, modes=%s" % (N, modes))

if __name__ == '__main__':
    for mode in modes:
        main_time = time.time()
        sim = base.AxSimulation(mode, 4.0, 3, processes, rest)
        res = sim.get_times_for_qsd(N, T, av_over, q_list, term)
        base.write_object_to_file(res, mode + '_times_N='+str(N)+'_av='+str(av_over)+'.data')
        log.info("computation of times for mode %s and N %s finished in %s min" %
                 (mode, N, round((time.time()-main_time)/60.0, 2)))
