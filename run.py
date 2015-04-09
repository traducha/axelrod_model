#-*- coding: utf-8 -*-

import time
from matplotlib import pyplot as plt
import logging as log
from multiprocessing import Pool
import sys
from base import *

def loop_over_q():
    """This function makes several things. Goal is to visualize
    some behavior of graphs in simulation. Function uses multiprocessing
    to run 4 processes at once. It plots switches in time, writes graphs to file
    and writes clusters vs. q to file.
    """
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    switch_function = switch_connection_BA
    N = 500
    av_k = 4.0
    f = 3
    clusters = {'q': [], 's': []}
    times = 20000000
    q_list = [3] + [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))][::3] + [1300, 9000]
    q_list = [(q_list[i], q_list[i+1], q_list[i+2], q_list[i+3]) for i in range(0, len(q_list), 4)]
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
            
            g.write_pickle('OUT/graph_N='+str(N)+'_q='+str(q[j])+'_T='+str(times))
            clusters['s'].append(len(g.clusters()[0]) * 1.0 / N)
            clusters['q'].append(q[j])
            
            plt.plot(x[::1000], y[::1000])
            plt.title("Network with N = %s nodes, f = %s, q = %s" % (N, f, q[j]))
            plt.xlabel('time step')
            plt.ylabel('total number of switches')
            plt.savefig("OUT/switches_N="+str(N)+"_q="+str(q[j])+".png", format="png")
            plt.clf()
        log.info("%s percent of algorithm executed" % round((100.0 * (q_list.index(q) + 1.0) / len(q_list)), 1) )
        
    write_clusters_to_file(clusters, name='OUT/clusters.txt')
    return True

def plot_sd_vs_q(name):
    """Use in interacive mode to plot results.
    @param name: name of file with data
    """
    r = read_object_from_file(name)
    plt.scatter(r['q'], r['s'], color='blue')
    plt.scatter(r['q'], r['d'], color='red')
    plt.xlim([1, 10000])
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.show()
    return True

def watch_one_graph(g, T):
    """This function runs simulation for one graph
    and saves largest component and domain for every time step.
    @param g: graph to start with
    @param T: number of time steps
    @return: dictionary with lists to plot
    """
    N = len(g.vs())
    res = {'t': [], 's': [], 'd': []}
    for t in range(T):
        g = basic_algorithm(g, 3, 1)
        if t % 100000 == 0:
            res['t'].append(t)
            res['s'].append(get_largest_component(g) * 1.0 / N)
            res['d'].append(get_largest_domain(g) * 1.0 / N)
            print t, is_switch_possible(g)
    write_object_to_file(res, 'play_in_time_N='+str(N)+'.data')
    plt.plot(res['t'], res['s'], color='blue')
    #plt.plot(res['t'], res['d'], color='red')
    plt.title("Largest component and domain in time, N = %s" % N)
    plt.xlabel('time step')
    plt.ylabel('largest component/domain')
    plt.savefig('play_in_time_N='+str(N)+'.png', format="png")
    return res

def get_data_for_qsd(N, T, av_over_q, q_list, processes):
    """Function with loop over q to get data for plots.
    @param N: number of nodes in graph
    @param T: number of time steps in base algorithm
    @param av_over_q: number of repetitions for one q
    @param q_list: list of q's values to iterate over
    @param processes: number of parallel processes
    @return dict: dictionary with lists of q, components and domains
    """
    q_list.sort()
    res = {'q': [], 's': [], 'd': []}
    for q in q_list:
        start_time = time.time()
        comp, dom = get_component(N, q, T, av_over_q, processes=processes)
        res['q'].append(q)
        res['s'].append(comp)
        res['d'].append(dom)
        log.info("computing components and domains for q = %s finished in %s minutes" % (q, round((time.time()-start_time)/60.0, 2)))
    return res

def main():
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    
    if '-p' in sys.argv:
        processes = int(sys.argv[sys.argv.index('-p')+1])
    else:
        processes = 1
        
    if processes == 1:
        get_component = get_average_component_for_q
    else:
        get_component = get_average_component_for_q_multi
    
    if '-m' in sys.argv:
        mode = sys.argv[sys.argv.index('-m')+1]
        if mode == 'BA':
            switch_function = switch_connection_BA
        else:
            switch_function = switch_connection_while
    else:
        mode = 'normal'
        switch_function = switch_connection_while
    
    N = 500
    av_q = 100
    T = 1200000
    
    main_time = time.time()
    q_list = [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))] #51 points in log scale
    res = get_data_for_qsd(N, T, av_q, q_list, processes=processes)
    write_object_to_file(res, 'res_N='+str(N)+'_q_times_'+str(av_q)+'_mode='+mode+'.data')
    log.info("main function executed in %s minutes" % round((time.time()-main_time)/60.0, 2))
    return

if __name__ == "__main__":
    #main()
    #loop_over_q()
    g = read_graph_from_file('OUT/graph_N=500_q=243_T=20000000')
    print is_static(g)
    print is_switch_possible(g)
    watch_one_graph(g, 10000000)
    
