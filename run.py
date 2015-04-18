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
    N = 500
    Sim = AxSimulation(mode=2, av_k=4.0, f=3)
    clusters = {'q': [], 's': []}
    times = 20000000
    q_list = [3] + [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))][::3] + [1300, 9000]
    q_list = [(q_list[i], q_list[i+1], q_list[i+2], q_list[i+3]) for i in range(0, len(q_list), 4)]
    for q in q_list:
        start_time = time.time()
        g1 = AxGraph.random_graph_with_attrs(N, Sim.av_k, Sim.f, q[0])
        g2 = AxGraph.random_graph_with_attrs(N, Sim.av_k, Sim.f, q[1])
        g3 = AxGraph.random_graph_with_attrs(N, Sim.av_k, Sim.f, q[2])
        g4 = AxGraph.random_graph_with_attrs(N, Sim.av_k, Sim.f, q[3])
        pool_agrs = [[g1, times], [g2, times], [g3, times], [g4, times]]
        
        pool = Pool(processes=4)
        res = pool.map_async(Sim.func_star, pool_agrs)
        pool.close()
        pool.join()
        pool.terminate()
        result = res.get()
        
        for j in range(4):
            g, x, y = result[j]
            log.info("algorithm for q = %s executed in %s seconds" % (q[j], round((time.time() - start_time), 4)))
            
            g.write_pickle('OUT/graph_N='+str(N)+'_q='+str(q[j])+'_T='+str(times))
            clusters['s'].append(len(g.clusters()[0]) * 1.0 / N)
            clusters['q'].append(q[j])
            
            plt.plot(x[::1000], y[::1000])
            plt.title("Network with N = %s nodes, f = %s, q = %s" % (N, Sim.f, q[j]))
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
    Sim = AxSimulation(2, 4.0, 3)
    res = {'t': [], 's': [], 'd': []}
    for t in range(T):
        g = Sim.basic_algorithm(g, 1)
        if t % 100000 == 0:
            res['t'].append(t)
            res['s'].append(g.get_largest_component() * 1.0 / N)
            res['d'].append(g.get_largest_domain() * 1.0 / N)
            print t, g.is_switch_possible()
    write_object_to_file(res, 'play_in_time_N='+str(N)+'.data')
    plt.plot(res['t'], res['s'], color='blue')
    #plt.plot(res['t'], res['d'], color='red')
    plt.title("Largest component and domain in time, N = %s" % N)
    plt.xlabel('time step')
    plt.ylabel('largest component/domain')
    plt.savefig('play_in_time_N='+str(N)+'.png', format="png")
    return res

def main(N=500, av_q=20, T=1000000):
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    # read initial arguments
    if '-p' in sys.argv:
        processes = int(sys.argv[sys.argv.index('-p')+1])
        log.info("%s child processes will be spawn" % processes)
    else:
        raise Exception("Use switch '-p' to define number of processes")
    
    if '-m' in sys.argv:
        mode = sys.argv[sys.argv.index('-m')+1]
        log.info("mode of simulation is: %s" % mode)
    else:
        raise Exception("Use switch '-m' to define mode of simulation")
    
    if '--rest' in sys.argv:
        rest = sys.argv[sys.argv.index('--rest')+1]
        rest = [float(i) for i in rest.split('-')]
        log.info("rest mode is: %s" % rest)
    else:
        rest = []
    # set simulation parameters
    q_list = [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))] #51 points in log scale
    simulation = AxSimulation(mode, 4.0, 3, processes, rest)
    # run simulation and save results
    main_time = time.time()
    res = simulation.get_data_for_qsd(N, T, av_q, q_list)
    write_object_to_file(res, 'OUT/res_N='+str(N)+'_q_times_'+str(av_q)+'_mode='+mode+'.data')
    log.info("main function executed in %s minutes" % round((time.time()-main_time)/60.0, 2))
    return

if __name__ == "__main__":
    #main(N=500, av_q=4, T=200000)

    q_list = [2, 5, 10, 20]
    simulation = AxSimulation('BA', 4.0, 3, 4, [])
    res = simulation.watch_many_graphs(500, 1000000, q_list)
    write_object_to_file(res, 'test.data')

    #loop_over_q()
#     g = read_graph_from_file('OUT/graph_N=500_q=243_T=20000000')
#     print g.is_static()
#     print g.is_switch_possible()
#     watch_one_graph(g, 10000000)
    
#     N = 500
#     av_q = 100
#     T = 1200000






# rysowanie
# a = plt.plot(r['q'], r['s'], color='blue')
# b = plt.scatter(r['q'], r['d'], color='red')
# plt.legend([a, b], ['Largest component', 'Largest domain'])
# plt.ylabel('fraction of # of nodes')
# plt.xlabel('q')
# plt.title('Network with 500 nodes, # of attrs f = 3.\nResults averaged over 400 realizations.')
# plt.plot(r['q'], r['s'], color='blue')
# plt.xlim([1, 10000])
# plt.ylim([0, 1])
# plt.xscale('log')
# plt.show()
# plt.clf()