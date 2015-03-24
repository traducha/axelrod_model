#-*- coding: utf-8 -*-

import time
import logging as log
from multiprocessing import Pool
import igraph as ig
import random
import numpy as np
from numpy.random import random as rand
from matplotlib import pyplot as plt

def write_clusters_to_file(clusters, name):
    file = open(name, 'w')
    file.writelines([str(clusters['q']).replace('[', '').replace(']', ''), '\n', str(clusters['s']).replace('[', '').replace(']', '')])
    file.close()
    return True

def red_clusters_from_file(name):
    file = open(name, 'r')
    q = file.readline()
    s = file.readline()
    file.close()
    q = [int(i) for i in q.split(', ')]
    s = [float(i) for i in s.split(', ')]
    return {'q': q, 's': s}

def random_graph_with_attrs(N=2500, av_k=4.0, f=3, q=2):
    """Creating random graph and generating random attributes.
    @param N: number of nodes in the graph
    @param av_k: expected average degree of the graph
    @param f: number of node's attributes
    @param q: number of possible values of an attribute
    @return: igraph.Graph object
    """
    p = av_k / (N - 1.0)
    g = ig.Graph.Erdos_Renyi(N, p)
    g.vs()["f"] = np.int_(rand((N, f))*q)
    log.info("initial average degree was: k = %s" % (np.sum(g.degree()) * 1.0 / N))
    return g

def switch_connection(g, index, del_index, n, neigs):
    """
    """
    g.delete_edges((index, del_index))
    new_neig = random.choice(list(set(range(n)).difference(neigs.append(index))))
    g.add_edges([(index, new_neig)])
    return g

def basic_algorithm(g, f, T):
    """
    """
    n = len(g.vs())
    switches_sum = [0]
    for i in range(T):
        #get one node and randomly select one of it's neighbors
        index = int(rand()*n)
        neigs = g.neighbors(index)
        if not neigs:
            switches_sum.append(switches_sum[-1])
            continue
        neig_index = random.choice(neigs)
        #compare attributes of two nodes
        vertex_attrs = g.vs(index)["f"][0]
        neighbor_attrs = g.vs(neig_index)["f"][0]
        m = np.count_nonzero((vertex_attrs == neighbor_attrs))
        #decide what to do according to common attributes
        if m == 0:
            switch_connection(g, index, neig_index, n, neigs)
            switches_sum.append(switches_sum[-1]+1)
        elif m != f and rand() < m*1.0/f:
            change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
            vertex_attrs[change_attr] = neighbor_attrs[change_attr]
            switches_sum.append(switches_sum[-1])
        else:
            switches_sum.append(switches_sum[-1])
    return g, range(T+1), switches_sum

def func_star(chain):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return basic_algorithm(*chain)

def loop_over_q():
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    N = 500
    av_k = 4.0
    f = 3
    clusters = {'q': [], 's': []}
    times = 10000000
    q_list = [[1, 2, 4, 8], [10, 20, 40, 80],\
        [100, 200, 400, 800], [1000, 700, 900, 500], [300, 85, 87, 83], [30, 50, 90, 95]] #range(700, 1000, 50) + range(1000, 4000, 200)
    
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

if __name__ == "__main__":
    main_time = time.time()
    loop_over_q()
    log.info("main() function executed in %s seconds" % (time.time() - main_time))


#reading g = ig.Graph.Read_Pickle('dupa')
#TODO:
#def check_clusters_homogenity(g):
#    a = g.clusters()
#    return g.clusters()
"""if i > 1000000:
        b = 0
        for sub_list in g.clusters():
            stop = 1
            sub = g.induced_subgraph(sub_list)
            attrs = sub.vs()["f"]
            for d in range(len(attrs)-1):
                if not all(attrs[d] == attrs[d+1]):
                    print "NOT"
                    stop = 0
                    break
            if stop:
                print "OK"
                b = 1
            else:
                break
        if b:
            print i
            break"""
