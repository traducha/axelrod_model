#-*- coding: utf-8 -*-

import igraph as ig
import logging as log
import random
import numpy as np
from numpy.random import random as rand
from multiprocessing import Pool
import pickle

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

def write_object_to_file(obj, name):
    """Writing objects to file.
    @param obj: object to write
    @param name: name of file to open/create and write into
    """
    f = open(name, 'wb')
    pickle.dump(obj, f)
    f.close()
    return True

def read_object_from_file(name):
    """Reading objects from files.
    @param name: name of the file
    @return: object that was in file
    """
    f = open(name, 'rb')
    res = pickle.load(f)
    f.close()
    return res

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
    #log.info("initial average degree was: k = %s" % (np.sum(g.degree()) * 1.0 / N))
    return g

def get_largest_component(g):
    """Returns number of nodes in largest component of graph.
    Its easy to change this function to return also number of components.
    @param g: graph to work with
    @return: largest component of g
    """
    return len(g.clusters()[0])

def get_largest_domain(g):
    """Returns number of nodes in largest domain of graph.
    Its easy to change this function to return also number of domains.
    @param g: graph to work with
    @return: largest domain of g
    """
    domains = {}
    uniq = {}
    for i, attrs in enumerate(g.vs()["f"]):
        for key, value in uniq.items():
            if all(attrs == value):
                domains[key] += 1
                break
        else:
            uniq[i] = attrs
            domains[i] = 1
    return max(domains.values())

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
    @param neigs: ids of neighbors of index
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    while 1:
        new_neig = random.randint(0,n-1)
        if new_neig not in neigs and new_neig != index:
            break
    g.add_edges([(index, new_neig)])
    return g

def switch_connection_BA(g, index, del_index, n, neigs):
    """This function switches connection for given node
    without doubling any connection. Probability is proportional
    to degree of a node.
    @param g: graph to work on
    @param index: id of main node
    @param del_index: id of node to disconnect
    @param n: number of nodes in g
    @param neigs: ids of neighbors of index
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    edges = len(g.get_edgelist()) #TODO: what is faster: g.es() or g.get_edgelist()?
    while 1:
        new_neig = g.get_edgelist()[random.randint(0, edges-1)][random.randint(0, 1)]
        if new_neig not in neigs and new_neig != index:
            break
    g.add_edges([(index, new_neig)])
    return g

def switch_function():
    """This function has to be overwrite to put in base_algorithm()
    proper function for switches. Idea of this function is
    to avoid unnecessary if/else in loop.
    """
    raise Exception("Function 'switch_fuction' has to be overwrite by one of real switching functions.\n" +
                    "Set 'switch_function = switch_connection_while' for example")
    return False
switch_function = switch_connection_BA

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
            switch_function(g, index, neig_index, n, neigs)
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
            switch_function(g, index, neig_index, n, neigs)
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

def get_average_component_for_q(N, q, T, av_over, processes=1):
    """This function calls base_algorithm for av_over times
    and computes average largest component and domain.
    @param N: number of nodes in graph
    @param q: number of possible values of node's attributes
    @param T: number of time steps for base_algorithm
    @param av_over: number of base_algorithm executions
    @param processes: does't matter here
    @return (float, float): average largest component and domain
    """
    biggest_clusters = []
    biggest_domain = []
    for i in range(av_over):
        g = random_graph_with_attrs(N, 4.0, 3, q)
        g = basic_algorithm(g, 3, T)
        biggest_clusters.append(get_largest_component(g) * 1.0 / N)
        biggest_domain.append(get_largest_domain(g) * 1.0 / N)
    return np.sum(biggest_clusters) * 1.0 / av_over, np.sum(biggest_domain) * 1.0 / av_over

def get_average_component_for_q_multi(N, q, T, av_over, processes):
    """This function calls base_algorithm for av_over times
    and computes average largest component and domain. It uses multiprocessing
    for spawning child processes.
    @param N: number of nodes in graph
    @param q: number of possible values of node's attributes
    @param T: number of time steps for base_algorithm
    @param av_over: number of base_algorithm executions
    @param processes: number of parallel processes
    @return (float, float): average largest component and domain
    """
    biggest_clusters = []
    biggest_domain = []
    
    def append_result(res):
        """""This function is called from the main process
        to append results to lists."""
        biggest_clusters.append(get_largest_component(res) * 1.0 / N)
        biggest_domain.append(get_largest_domain(res) * 1.0 / N)
        return True
    
    pool = Pool(processes=processes)
    for i in range(av_over):
        g = random_graph_with_attrs(N, 4.0, 3, q)
        pool.apply_async(basic_algorithm, args=(g, 3, T), callback=append_result)
    pool.close()
    pool.join()
    pool.terminate()
    return np.sum(biggest_clusters) * 1.0 / av_over, np.sum(biggest_domain) * 1.0 / av_over

#reading g = ig.Graph.Read_Pickle('name')