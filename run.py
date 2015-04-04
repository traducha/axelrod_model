#-*- coding: utf-8 -*-

import time
from matplotlib import pyplot as plt
import logging as log
from multiprocessing import Pool
from base import *

def loop_over_q():
    """This function makes several things. Goal is to visualize
    some behavior of graphs in simulation. Function uses multiprocessing
    to run 4 processes at once. It plots switches in time, writes graphs to file
    and writes clusters vs. q to file.
    """
    N = 500
    av_k = 4.0
    f = 3
    clusters = {'q': [], 's': []}
    times = 1000000
    q_list = [[20, 40, 60, 80]]#, [10, 12, 15, 20], [25, 30, 35, 40], [45, 50, 55, 60], [65, 70, 75, 80], [85, 90, 95, 100]] #range(700, 1000, 50) + range(1000, 4000, 200)
    #q_list += [[110, 120, 150, 200], [250, 300, 350, 400], [450, 500, 550, 600], [310, 320, 330, 340], [360, 370, 380, 390]]
    for q in q_list:
        start_time = time.time()
        g1 = random_graph_with_attrs(N, av_k, f, q[0])
        g2 = random_graph_with_attrs(N, av_k, f, q[1])
        g3 = random_graph_with_attrs(N, av_k, f, q[2])
        g4 = random_graph_with_attrs(N, av_k, f, q[3])
        pool_agrs = [[g1, f, times], [g2, f, times], [g3, f, times], [g4, f, times]]
        
        pool = Pool(processes=4)
        res = pool.map_async(func_star, pool_agrs)
        pool.close()
        pool.join()
        result = res.get()
        
        for j in range(4):
            g, x, y = result[j]
            log.info("algorithm for q = %s executed in %s seconds" % (q[j], round((time.time() - start_time), 4)))
            
            g.write_pickle('graph_N='+str(N)+'_q='+str(q[j])+'_T='+str(times))
            clusters['s'].append(len(g.clusters()[0]) * 1.0 / N)
            clusters['q'].append(q[j])
            
            plt.plot(x[::1000], y[::1000])
            plt.title("Network with N = %s nodes, f = %s, q = %s" % (N, f, q[j]))
            plt.xlabel('time step')
            plt.ylabel('total number of switches')
            plt.savefig("switches_N="+str(N)+"_q="+str(q[j])+".png", format="png")
            plt.clf()
        log.info("%s percent of algorithm executed" % round((100.0 * (q_list.index(q) + 1.0) / len(q_list)), 1) )
        
    write_clusters_to_file(clusters, name='clusters.txt')
    return True



def get_average_component_for_q(N, q, T, av_over):
    """This function calls base_algorithm for av_over times
    and computes average largest component and domain.
    @param N: number of nodes in graph
    @param q: number of possible values of node's attributes
    @param T: number of time steps for base_algorithm
    @param av_over: number of base_algorithm executions
    @return: average largest component and domain
    """
    biggest_clusters = []
    biggest_domain = []
    for i in range(av_over):
        g = random_graph_with_attrs(N, av_k=4.0, f=3, q)
        g = basic_algorithm(g, f=3, T)
        biggest_clusters.append(oblicz najw komponent g)
        biggest_domain.append(oblicz najw domenę g)
    return np.sum(biggest_clusters) * 1.0 / av_over, np.sum(biggest_domain) * 1.0 / av_over

if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    main_time = time.time()
    loop_over_q()
    log.info("main() function executed in %s seconds" % (time.time() - main_time))

