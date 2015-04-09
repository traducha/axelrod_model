#-*- coding: utf-8 -*-

import igraph as ig
import logging as log
import random
import numpy as np
from numpy.random import random as rand
from multiprocessing import Pool
import pickle

#########################################################################
# Static methods for writing objects to and reading from files.         #
#########################################################################

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

def read_graph_from_file(name):
    """This function reads graph from pickle file.
    @param name: name of the file
    @return: Graph object
    """
    return AxGraph.load(name)

#########################################################################
#                Main class for graphs in this module.                  #
#########################################################################

class AxGraph(ig.Graph):
    """This class extends igraph.Graph by several methods
    useful in simulating axelrod's model on coevolving networks.
    """
    
    @classmethod
    def random_graph_with_attrs(cls, N=500, av_k=4.0, f=3, q=2):
        """Creating random graph and generating random attributes.
        Average degree will be approximately equal av_k.
        @param N: number of nodes in the graph
        @param av_k: expected average degree of the graph
        @param f: number of node's attributes
        @param q: number of possible values of an attribute
        @return: igraph.Graph object
        """
        p = av_k * 1.0 / (N - 1.0)
        g = cls.Erdos_Renyi(N, p)
        g.vs()["f"] = np.int_(rand((N, f))*q)
        #log.info("initial average degree was: k = %s" % (np.sum(g.degree()) * 1.0 / N))
        return g
    
    @classmethod
    def load(cls, name):
        """This method reads graph from pickle file.
        @param name: name of the file
        @return: Graph object
        """
        return cls.Read_Pickle(name)

    def is_static(self):
        """This method finds out if switch or interaction is ever possible,
        i.e. if there is connected pair of nodes with not all
        attributes equal.
        @param g: graph
        @return boolean: True if graph is static
        """
        for pair in self.get_edgelist():
            if not all(self.vs(pair[0])["f"][0] == self.vs(pair[1])["f"][0]):
                return False
        return True

    def is_switch_possible(self):
        """This method finds out if switch is ever possible,
        i.e. if there is connected pair of nodes with all
        attributes different.
        @param g: graph
        @return boolean: True if switch is possible
        """
        for pair in self.get_edgelist():
            if all(self.vs(pair[0])["f"][0] != self.vs(pair[1])["f"][0]):
                return True
        return False

    def get_largest_component(self):
        """Returns number of nodes in largest component of graph.
        Its easy to change this function to return also number of components.
        @param g: graph to work with
        @return: largest component of g
        """
        return len(max(self.clusters(), key = lambda clust: len(clust)))

    def get_largest_domain(self):
        """Returns number of nodes in largest domain of graph.
        Its easy to change this function to return also number of domains.
        @param g: graph to work with
        @return: largest domain of g
        """
        domains = {}
        uniq = {}
        for i, attrs in enumerate(self.vs()["f"]):
            for key, value in uniq.items():
                if all(attrs == value):
                    domains[key] += 1
                    break
            else:
                uniq[i] = attrs
                domains[i] = 1
        return max(domains.values())
    
#########################################################################
# Class holding functions for simulations and values of parameters.     #
#########################################################################

class AxSimulation():
    """Class holding methods for simulating axelrod's model
    on coevolving networks. It keeps parameters of simulation
    and depending on their values provides proper behavior.
    Simulation can be run as one process or split into
    several processes to speed up a bit.
    """
    
    def __init__(self, mode, av_k, f):
        """Set up parameters for simulation.
        @param mode: mode of simulation, switching behavior depends on it
        @param av_k: average degree of nodes
        @param f: number of attributes (every attribute can have q different values)
        """
        if mode in ['normal', 'random', 1]:
            self.switch_function = self.switch_connection_while
        elif mode in ['BA', 2]:
            self.switch_function = self.switch_connection_BA
        else:
            raise ValueError("Invalid mode of simulation! Choose one of 'normal', 'random', 'BA', 1 or 2. %s was given." % mode)
        
        if not isinstance(av_k, (int, float)):
            raise TypeError("Second argument of AxSimulation have to be one of int, float! %s was given." % type(av_k))
        
        if not isinstance(f, int):
            raise TypeError("Third argument of AxSimulation have to be int! %s was given." % type(f))
        
        self.mode = mode
        self.av_k = av_k
        self.f = f

    def switch_connection(self, g, index, del_index, n, neigs):
        """This method switches connection for given node.
        For big graphs and relatively small neighborhood function
        switch_connection_while() is faster so this method is not
        used at the moment.
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

    def switch_connection_while(self, g, index, del_index, n, neigs):
        """This method switches connection for given node
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

    def switch_connection_BA(self, g, index, del_index, n, neigs):
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
        edges = len(g.es())  # g.es() is faster than g.get_edgelist()
        while 1:
            new_neig = g.get_edgelist()[random.randint(0, edges-1)][random.randint(0, 1)]
            if new_neig not in neigs and new_neig != index:
                break
        g.add_edges([(index, new_neig)])
        return g

    def basic_algorithm(self, g, T):
        """This is the basic algorithm of axelrod's coevolving network.
        @param g: graph to work on
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
                self.switch_function(g, index, neig_index, n, neigs)
            elif m != self.f and rand() < m*1.0/self.f:
                change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
                vertex_attrs[change_attr] = neighbor_attrs[change_attr]
        return g

    def basic_algorithm_count_switches(self, g, T):
        """Copy of basic algorithm which counts number of switches
        from the beginning of simulation. Separated from basic_algorithm()
        to keep speed of that function.
        @param g: graph to work on
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
                self.switch_function(g, index, neig_index, n, neigs)
                switches_sum.append(switches_sum[-1]+1)
            elif m != self.f and rand() < m*1.0/self.f:
                change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
                vertex_attrs[change_attr] = neighbor_attrs[change_attr]
                switches_sum.append(switches_sum[-1])
            else:
                switches_sum.append(switches_sum[-1])
        return g, range(T+1)[::20], switches_sum[::20]

    def func_star(self, chain):
        """Converts `f([1,2])` to `f(1,2)` call.
        It's necessary when using Pool.map_async() function"""
        return self.basic_algorithm_count_switches(*chain)

    def get_average_component_for_q(self, N, q, T, av_over, processes=1):
        """This method calls base_algorithm for av_over times
        and computes average largest component and domain.
        @param N: number of nodes in graph
        @param q: number of possible values of node's attributes
        @param T: number of time steps for base_algorithm
        @param av_over: number of base_algorithm executions
        @param processes: does't matter here,
        left to call the same way as get_average_component_for_q_multi
        @return (float, float): average largest component and domain
        """
        biggest_clusters = []
        biggest_domain = []
        for i in range(av_over):
            g = AxGraph.random_graph_with_attrs(N, self.av_k, self.f, q)
            g = self.basic_algorithm(g, T)
            biggest_clusters.append(g.get_largest_component() * 1.0 / N)
            biggest_domain.append(g.get_largest_domain() * 1.0 / N)
        return np.sum(biggest_clusters) * 1.0 / av_over, np.sum(biggest_domain) * 1.0 / av_over

    def get_average_component_for_q_multi(self, N, q, T, av_over, processes):
        """This method calls base_algorithm for av_over times
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
        
        def append_result(res_g):
            """This function is called from the main process
            to append results to lists.
            @param res_g: object of AxGraph class
            """
            biggest_clusters.append(res_g.get_largest_component() * 1.0 / N)
            biggest_domain.append(res_g.get_largest_domain() * 1.0 / N)
            return True
        
        pool = Pool(processes=processes)
        for i in range(av_over):
            g = AxGraph.random_graph_with_attrs(N, self.av_k, self.f, q)
            pool.apply_async(self.basic_algorithm, args=(g, T), callback=append_result)
        pool.close()
        pool.join()
        pool.terminate()
        return np.sum(biggest_clusters) * 1.0 / av_over, np.sum(biggest_domain) * 1.0 / av_over

if __name__ == '__main__':
    x = AxGraph.random_graph_with_attrs(500, 4, 3, 30)
    print x
    print type(x)