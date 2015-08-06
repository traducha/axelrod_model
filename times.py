#-*- coding: utf-8 -*-

import logging as log
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
mode = 'normal'
N = 500
av_over = 16
q_list = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
T = 3000000
term = 50000
processes = 8
rest = [10.0, 0.5]
log.info("algorithm computing times has started for N=%s, mode=%s" % (N, mode))

if __name__ == '__main__':
    main_time = time.time()
    sim = base.AxSimulation(mode, 4.0, 3, processes, rest)
    res = sim.get_times_for_qsd(N, T, av_over, q_list, term)
    base.write_object_to_file(res, 'times_N='+str(N)+'_av='+str(av_over)+'.data')
    log.info("computation of times for mode %s and N %s finished in %s min" %
             (mode, N, round((time.time()-main_time)/60.0, 2)))
