#-*- coding: utf-8 -*-

import igraph as ig
import random
import numpy as np
from numpy.random import random as rand

def write_clusters_to_file(clusters, name):
    """Old function to write clusters vs. q data to file. Better use pickle.
    @param clusters: dictionary with two keys
    @param name: name of file to create
    """
    file = open(name, 'w')
    file.writelines([str(clusters['q']).replace('[', '').replace(']', ''), '\n', str(clusters['s']).replace('[', '').replace(']', '')])
    file.close()
    return True

def read_clusters_from_file(name):
    """Old function to read clusters vs. q data from file made in 
    write_clusters_to_file() function. Better use pickle.
    @param name: name of file to read from
    """
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
    """This function switches connection for given node.
    For big graphs and relatively small neighborhood function
    switch_connection_while() is faster.
    @param g: graph to work on
    @param index: id of main node
    @param del_index: id of node to disconnect
    @param n: number of nodes in g
    @param neigs: ids of neghbors of index
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    new_neig = random.choice(list(set(range(n)).difference(neigs + [index])))
    g.add_edges([(index, new_neig)])
    return g

def switch_connection_while(g, index, del_index, n, neigs):
    """This function switches connection for given node
    without doubling any connection. Every node has the same probability.
    @param g: graph to work on
    @param index: id of main node
    @param del_index: id of node to disconnect
    @param n: number of nodes in g
    @param neigs: ids of neghbors of index
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    while 1:
        new_neig = random.randint(0,n-1)
        try:
            g.es.find(_between=((index,), (new_neig,)))
        except ValueError:
            break
    g.add_edges([(index, new_neig)])
    return g

def switch_connection_BA(g, index, del_index, n, edges):
    """This function switches connection for given node
    without doubling any connection. Probability is proportional
    to degree of a node.
    @param g: graph to work on
    @param index: id of main node
    @param del_index: id of node to disconnect
    @param n: number of nodes in g
    @param edges: number of edges in g
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    while 1:
        new_neig = g.get_edgelist()[random.randint(0, edges-1)][random.randint(0, 1)]
        try:
            g.es.find(_between=((index,), (new_neig,)))
        except ValueError:
            break
    g.add_edges([(index, new_neig)])
    return g

def basic_algorithm(g, f, T):
    """This is the basic algorithm of coevolving network.
    @param g: graph to work on
    @param f: number of attributes of nodes
    @param T: number of time steps
    @return: g after applying algorithm
    """
    n = len(g.vs())
    for i in range(T):
        #get one node and randomly select one of it's neighbors
        index = int(rand()*n)
        neigs = g.neighbors(index)
        if not neigs:
            continue
        neig_index = random.choice(neigs)
        #compare attributes of two nodes
        vertex_attrs = g.vs(index)["f"][0]
        neighbor_attrs = g.vs(neig_index)["f"][0]
        m = np.count_nonzero((vertex_attrs == neighbor_attrs))
        #decide what to do according to common attributes
        if m == 0:
            switch_connection_while(g, index, neig_index, n, neigs)
        elif m != f and rand() < m*1.0/f:
            change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
            vertex_attrs[change_attr] = neighbor_attrs[change_attr]
    return g

def basic_algorithm_count_switches(g, f, T):
    """Copy of basic algorithm which counts number of switches
    from the beginning of simulation. Separated from basic_algorithm()
    to keep speed of that function.
    @param g: graph to work on
    @param f: number of attributes of nodes
    @param T: number of time steps
    @return (graph, list, list): g after applying algorithm,
    list with number of time steps, list with sum of switches from the beginning
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
            switch_connection_while(g, index, neig_index, n, neigs)
            switches_sum.append(switches_sum[-1]+1)
        elif m != f and rand() < m*1.0/f:
            change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
            vertex_attrs[change_attr] = neighbor_attrs[change_attr]
            switches_sum.append(switches_sum[-1])
        else:
            switches_sum.append(switches_sum[-1])
    return g, range(T+1), switches_sum

def func_star(chain):
    """Converts `f([1,2])` to `f(1,2)` call.
    It's necessary when using Pool.map_async() function"""
    return basic_algorithm_count_switches(*chain)


#reading g = ig.Graph.Read_Pickle('name')
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
