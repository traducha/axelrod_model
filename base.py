#-*- coding: utf-8 -*-

import time
import logging as log
import igraph as ig
import random
import numpy as np
from numpy.random import random as rand
from matplotlib import pyplot as plt

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
    new_neig = random.choice(list(set(range(n)).difference(neigs)))
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

def main():
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    N = 100
    av_k = 4.0
    f = 3
    times = 100
    q_list = range(2, 10, 2) + range(10, 100, 5) + range(100, 1000, 50) + range(1000, 4000, 200)
    for q in q_list:
        start_time = time.time()
        g = random_graph_with_attrs(N, av_k, f, q)
        _, x, y = basic_algorithm(g, f, times)
        log.info("algorithm for q = %s executed in %s seconds" % (q, (time.time() - start_time)))
        
        plt.plot(x, y)
        plt.title("Network with N = %s nodes, f = %s, q = %s" % (N, f, q))
        plt.xlabel('time step')
        plt.ylabel('total number of switches')
        plt.savefig("switches_N="+str(N)+"_q="+str(q)+".png", format="png")
        log.info("%s\% of algorithm executed" % (q, (time.time() - start_time)))

if __name__ == "__main__":
    main_time = time.time()
    main()
    log.info("main() function executed in %s seconds" % (time.time() - main_time))


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