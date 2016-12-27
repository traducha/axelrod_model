# -*- coding: utf-8 -*-

import igraph as ig
import time
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

    def pickle(self, name):
        """Writes graph to file in pickle format.
        """
        return self.write_pickle(name)

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

    def is_component_static(self):
        """This method finds out if switch is ever possible,
        i.e. if there is connected pair of nodes with all
        attributes different.
        @param g: graph
        @return boolean: True if switch is possible
        """
        for pair in self.get_edgelist():
            if all(self.vs(pair[0])["f"][0] != self.vs(pair[1])["f"][0]):
                return False
        return True

    def is_domain_static(self):
        """
        """
        N = len(self.vs())
        for i in range(N):
            for j in range(i+1, N):
                if not (all(self.vs(i)["f"][0] == self.vs(j)["f"][0]) or all(self.vs(i)["f"][0] != self.vs(j)["f"][0])):
                    return False
        return True

    def what_is_static(self):
        """This method finds out whether domain or component
        can ever change again in coevolving axelrod's simulation.
        Condition for domain is not closed, but approximately acceptable.
        It also checks if whole graph is static.
        @return dict: True or False for 'domain', 'component' and 'all' keys.
        """
        res = {'domain': True, 'component': True, 'all': True}
        for i, j in self.get_edgelist():
            if res['domain']:
                if not (all(self.vs(i)["f"][0] == self.vs(j)["f"][0]) or all(self.vs(i)["f"][0] != self.vs(j)["f"][0])):
                    res['domain'] = False
            if res['component']:
                if all(self.vs(i)["f"][0] != self.vs(j)["f"][0]):
                    res['component'] = False
            if res['all']:
                if not all(self.vs(i)["f"][0] == self.vs(j)["f"][0]):
                    res['all'] = False
            if not (res['all'] or res['component'] or res['domain']):
                break
        return res
    
    # TODO True jeżeli wszystki możliwe niepołączone pary maja 0 wspólnych atr, a wszystkie połączone maja wszystkie takie same attr lub 0
    def is_dynamically_trapped(self):
        """This method finds out if graph is in dynamic equilibrium,
        i.e. there is no chance for interaction between two
        nodes ever, only edges keep switching.
        @param g: graph
        @return boolean: True if graph is trapped
        """
        N = len(self.vs())
        for i in range(N):
            for j in range(i+1, N):
                if not (all(self.vs(i)["f"][0] == self.vs(j)["f"][0]) or all(self.vs(i)["f"][0] != self.vs(j)["f"][0])):
                    return False
        return True

    def get_components(self):
        """Returns components of graph in from of dict of lists.
        Its lighter than AxGraph.clusters().
        """
        components = {}
        for i, comp in enumerate(self.clusters()):
            components[str(i)] = len(comp)
        return components

    def get_largest_component(self):
        """Returns number of nodes in largest component of graph.
        Its easy to change this function to return also number of components.
        """
        return len(max(self.clusters(), key = lambda clust: len(clust)))
    
    def get_number_of_components(self):
        """Returns number of components in graph.
        """
        return len(self.clusters())
    
    def get_domains(self):
        """Returns domains, number and sizes.
        """
        domains = {}
        for j, g in enumerate(self.clusters().subgraphs()):
            uniq = {}
            for i, attrs in enumerate(g.vs()["f"]):
                for key, value in uniq.items():
                    if all(attrs == value):
                        domains[key] += 1
                        break
                else:
                    uniq[str(j)+'_'+str(i)] = attrs
                    domains[str(j)+'_'+str(i)] = 1
        return domains

    def get_largest_domain(self):
        """Returns number of nodes in largest domain of graph.
        """
        domains = {}
        for j, g in enumerate(self.clusters().subgraphs()):
            uniq = {}
            for i, attrs in enumerate(g.vs()["f"]):
                for key, value in uniq.items():
                    if all(attrs == value):
                        domains[key] += 1
                        break
                else:
                    uniq[str(j)+'_'+str(i)] = attrs
                    domains[str(j)+'_'+str(i)] = 1
        return max(domains.values())
    
    def get_number_of_domains(self):
        """Returns number of domains in graph.
        """
        domains = {}
        for j, g in enumerate(self.clusters().subgraphs()):
            uniq = {}
            for i, attrs in enumerate(g.vs()["f"]):
                for key, value in uniq.items():
                    if all(attrs == value):
                        domains[key] += 1
                        break
                else:
                    uniq[str(j)+'_'+str(i)] = attrs
                    domains[str(j)+'_'+str(i)] = 1
        return len(domains)

    def get_number_of_domains_properly(self):
        """Returns number of domains in graph.
        """
        domains = 0
        used = []

        def find_more(used, dom, index):
            neigs = self.neighbors(index)
            for neig_index in neigs:
                    m = np.count_nonzero((self.vs(index)["f"][0] == self.vs(neig_index)["f"][0]))
                    if m == 3 and neig_index not in used:
                        dom.append(neig_index)
                        used.append(neig_index)
                        used, dom = find_more(used, dom, neig_index)
            return used, dom

        for i in range(len(self.vs())):
            if i not in used:
                used.append(i)
                neigs = self.neighbors(i)
                if not neigs:
                    domains += 1
                    continue
                dom = [i]
                used, dom = find_more(used, dom, i)
                domains += 1
        return domains


#########################################################################
# Functions used in running simulation on several processes.            #
# Multiprocessing can not send instance methods to child processes.     #
#########################################################################


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
    edges = len(g.es())  # g.es() is faster than g.get_edgelist()
    while 1:
        new_neig = g.get_edgelist()[random.randint(0, edges-1)][random.randint(0, 1)]
        if new_neig not in neigs and new_neig != index:
            break
    g.add_edges([(index, new_neig)])
    return g


def switch_connection_cluster(g, index, del_index, n, neigs):
    """
    @param g: graph to work on
    @param index: id of main node
    @param del_index: id of node to disconnect
    @param n: number of nodes in g
    @param neigs: ids of neighbors of index
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    switch_group = []
    new_neig = None
    for node in neigs:
        if node != del_index:
            switch_group += g.neighbors(node)
    if switch_group:
        switch_group = list(set(switch_group))
        random.shuffle(switch_group)
        for i in switch_group:
            if i not in neigs and i != index:
                new_neig = i
                break
    if not new_neig:
        while 1:
            new_neig = random.randint(0, n-1)
            if new_neig not in neigs and new_neig != index:
                break
    g.add_edges([(index, new_neig)])
    return g

    # different method not allowing network to break into parts
    # switch_group = g.neighbors(del_index)
    # random.shuffle(switch_group)
    # for i in switch_group:
    #     if i not in neigs and i != index:
    #         g.delete_edges((index, del_index))
    #         g.add_edges([(index, i)])
    #         break
    # return g

__a = 1


def switch_connection_k_plus_a(g, index, del_index, n, neigs):
    """This function switches connection for given node
    without doubling any connection. Probability is proportional
    to degree of a node plus A.
    @param g: graph to work on
    @param index: id of main node
    @param del_index: id of node to disconnect
    @param n: number of nodes in g
    @param neigs: ids of neighbors of index
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    cumulative_degree_plus_a = np.cumsum(np.array(g.degree()) + __a)
    while 1:
        new_neig = np.searchsorted(cumulative_degree_plus_a, random.randint(1, cumulative_degree_plus_a[-1]))
        if new_neig not in neigs and new_neig != index:
            break
    g.add_edges([(index, new_neig)])
    return g


def switch_connection_k_plus_a2(g, index, del_index, n, neigs):
    """This function switches connection for given node
    without doubling any connection. Probability is proportional
    to degree of a node plus A.
    @param g: graph to work on
    @param index: id of main node
    @param del_index: id of node to disconnect
    @param n: number of nodes in g
    @param neigs: ids of neighbors of index
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    cumulative_degree_plus_a = np.cumsum((np.array(g.degree()) + __a)**2.0)
    while 1:
        new_neig = np.searchsorted(cumulative_degree_plus_a, random.randint(1, cumulative_degree_plus_a[-1]))
        if new_neig not in neigs and new_neig != index:
            break
    g.add_edges([(index, new_neig)])
    return g


def switch_connection_high_k_cluster(g, index, del_index, n, neigs):
    """
    @param g: graph to work on
    @param index: id of main node
    @param del_index: id of node to disconnect
    @param n: number of nodes in g
    @param neigs: ids of neighbors of index
    @return: graph g after switch
    """
    g.delete_edges((index, del_index))
    switch_group = []
    new_neig = None
    for node in neigs:
        if node != del_index:
            switch_group.extend(g.neighbors(node))
    switch_group = list(set(switch_group) - set(neigs) - {index})
    if switch_group:
        if len(switch_group) == 1:
            new_neig = switch_group[0]
        else:
            cumulative_degree_plus_a = np.cumsum((np.array(g.vs(switch_group).degree()) + __a) ** 2.0)
            new_neig = switch_group[np.searchsorted(cumulative_degree_plus_a,
                                                    random.randint(1, cumulative_degree_plus_a[-1]))]
    else:
        while 1:
            new_neig = random.randint(0, n - 1)
            if new_neig not in neigs and new_neig != index:
                break
    g.add_edges([(index, new_neig)])
    return g


SWITCH_MAP = {'1': switch_connection_while, '2': switch_connection_BA,
              1: switch_connection_while, 2: switch_connection_BA,
              'normal': switch_connection_while, 'BA': switch_connection_BA,
              'cluster': switch_connection_cluster, '3': switch_connection_cluster,
              3: switch_connection_cluster, 'k_plus_a': switch_connection_k_plus_a,
              '4': switch_connection_k_plus_a, 4: switch_connection_k_plus_a,
              'k_plus_a2': switch_connection_k_plus_a2,
              '5': switch_connection_k_plus_a2, 5: switch_connection_k_plus_a2,
              'high_k_cluster': switch_connection_high_k_cluster,
              6: switch_connection_high_k_cluster, '6': switch_connection_high_k_cluster}


def basic_algorithm_multi(mode, f, g, T):
    """This is the basic algorithm of axelrod's coevolving network.
    @param mode: mode of simulation, defines switching function
    @param f: number of attributes of nodes
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
            SWITCH_MAP[mode](g, index, neig_index, n, neigs)
        elif m != f and rand() < m*1.0/f:
            change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
            vertex_attrs[change_attr] = neighbor_attrs[change_attr]
        if i % 100000 == 0:
            if g.is_static():
                return g
    return g


def find_times_multi(mode, f, g, T, term):
    """
    @param mode: mode of simulation, defines switching function
    @param f: number of attributes of nodes
    @param g: graph to work on
    @param T: number of time steps
    @return: g after applying algorithm
    """
    res = {}
    last_statics = {'domain': False, 'component': False, 'all': False}
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
            SWITCH_MAP[mode](g, index, neig_index, n, neigs)
        elif m != f and rand() < m*1.0/f:
            change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
            vertex_attrs[change_attr] = neighbor_attrs[change_attr]
        if i > term and i % 500 == 0:
            statics = g.what_is_static()
            for key, value in statics.items():
                if value and not last_statics[key]:
                    res[key] = i
            last_statics = statics
            if statics['all'] and statics['component'] and statics['domain']:
                break
    if 'domain' not in res:
        res['domain'] = T
    if 'component' not in res:
        res['component'] = T
    if 'all' not in res:
        res['all'] = T
    return res


def find_times_multi_properly(mode, f, g, T, term=500000):
    """
    @param mode: mode of simulation, defines switching function
    @param f: number of attributes of nodes
    @param g: graph to work on
    @param T: number of time steps
    @return: g after applying algorithm
    """
    res = {}
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
            SWITCH_MAP[mode](g, index, neig_index, n, neigs)
            res['component'] = i
        elif m != f and rand() < m*1.0/f:
            change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
            vertex_attrs[change_attr] = neighbor_attrs[change_attr]
            res['domain'] = i
        if i > term:
            if i % 50000 == 0:
                if g.is_static():
                    break
    res['all'] = max(res['component'], res['domain'])
    return res


#########################################################################
# Class holding functions for simulations and values of parameters.     #
#########################################################################


class AxSimulation:
    """Class holding methods for simulating axelrod's model
    on coevolving networks. It keeps parameters of simulation
    and depending on their values provides proper behavior.
    Simulation can be run as one process or split into
    several processes to speed up a bit.
    """
    
    def __init__(self, mode, av_k, f, processes, rest):
        """Set up parameters for simulation.
        @param mode: mode of simulation, switching behavior depends on it
        @param av_k: average degree of nodes
        @param f: number of attributes (every attribute can have q different values)
        @param processes: number of parallel processes to spawn
        """
        # self.CLUSTER = 0
        # self.NOT = 0
        # self.switch_connection_while = switch_connection_while
        
        if mode in ['normal', '1', 1]:
            self.switch_function = self.switch_connection_while
        elif mode in ['BA', '2', 2]:
            self.switch_function = self.switch_connection_BA
        elif mode in ['cluster', '3', 3]:
            self.switch_function = self.switch_connection_cluster
        elif mode in ['k_plus_a', '4', 4]:
            self.switch_function = self.switch_connection_k_plus_a
            self.a = 1
        elif mode in ['k_plus_a2', '5', 5]:
            self.switch_function = self.switch_connection_k_plus_a2
            self.a = 1
        elif mode in ['high_k_cluster', '6', 6]:
            self.switch_function = self.switch_connection_high_k_cluster
            self.a = 1
        else:
            raise ValueError("Invalid mode of simulation! Choose one of 'normal', 'random', 'BA', 1 or 2. %s was given." % mode)
        
        if not isinstance(av_k, (int, float)):
            raise TypeError("Second argument of AxSimulation have to be one of int, float! %s was given." % type(av_k))
        
        if not isinstance(f, int):
            raise TypeError("Third argument of AxSimulation have to be int! %s was given." % type(f))
        
        if not isinstance(processes, int):
            raise TypeError("Fourth argument of AxSimulation have to be int! %s was given." % type(processes))
        
        if not isinstance(rest, list) or len(rest) not in [0, 2]:
            raise TypeError("Fifth argument of AxSimulation have to be list of length 2 or 0! %s was given." % rest)
        for i in rest:
            if not isinstance(i, (int, float)):
                raise TypeError("Fifth argument of AxSimulation have to be list of ints or floats! %s was given." % type(i))
        
        if processes < 1:
            raise ValueError("Number of processes must be grater than zero! %s was given." % processes)
        elif processes == 1:
            self.compute_comp_dom = self.get_average_component_for_q
        else:
            self.compute_comp_dom = self.get_average_component_for_q_multi
            
        log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
        
        self.mode = mode
        self.av_k = av_k
        self.f = f
        self.processes = processes
        self.rest = rest
        self._start_time = time.time()

    def set_a(self, a):
        self.a = a
        return
        
    def try_sleep(self):
        if self.rest and ((time.time() - self._start_time) >= (self.rest[0] * 3600)):
            log.info("Going to sleep for %s hours." % (self.rest[1]))
            time.sleep(self.rest[1] * 3600)
            self._start_time = time.time()
        return

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

    def switch_connection_cluster(self, g, index, del_index, n, neigs):
        """
        @param g: graph to work on
        @param index: id of main node
        @param del_index: id of node to disconnect
        @param n: number of nodes in g
        @param neigs: ids of neighbors of index
        @return: graph g after switch
        """
        g.delete_edges((index, del_index))
        switch_group = []
        new_neig = None
        for node in neigs:
            if node != del_index:
                switch_group += g.neighbors(node)
        if switch_group:
            switch_group = list(set(switch_group))
            random.shuffle(switch_group)
            for i in switch_group:
                if i not in neigs and i != index:
                    # self.CLUSTER += 1
                    new_neig = i
                    break
        if not new_neig:
            # self.NOT += 1
            # edges = len(g.es())  # g.es() is faster than g.get_edgelist()
            while 1:
                new_neig = random.randint(0, n-1)
                if new_neig not in neigs and new_neig != index:
                    break
        g.add_edges([(index, new_neig)])
        return g

    def switch_connection_k_plus_a(self, g, index, del_index, n, neigs):
        """This function switches connection for given node
        without doubling any connection. Probability is proportional
        to degree of a node plus A.
        @param g: graph to work on
        @param index: id of main node
        @param del_index: id of node to disconnect
        @param n: number of nodes in g
        @param neigs: ids of neighbors of index
        @return: graph g after switch
        """
        g.delete_edges((index, del_index))
        cumulative_degree_plus_a = np.cumsum(np.array(g.degree()) + self.a)
        while 1:
            new_neig = np.searchsorted(cumulative_degree_plus_a, random.randint(1, cumulative_degree_plus_a[-1]))
            if new_neig not in neigs and new_neig != index:
                break
        g.add_edges([(index, new_neig)])
        return g

    def switch_connection_k_plus_a2(self, g, index, del_index, n, neigs):
        """This function switches connection for given node
        without doubling any connection. Probability is proportional
        to square of degree of a node plus A.
        @param g: graph to work on
        @param index: id of main node
        @param del_index: id of node to disconnect
        @param n: number of nodes in g
        @param neigs: ids of neighbors of index
        @return: graph g after switch
        """
        g.delete_edges((index, del_index))
        cumulative_degree_plus_a = np.cumsum((np.array(g.degree()) + self.a)**2.0)
        while 1:
            new_neig = np.searchsorted(cumulative_degree_plus_a, random.randint(1, cumulative_degree_plus_a[-1]))
            if new_neig not in neigs and new_neig != index:
                break
        g.add_edges([(index, new_neig)])
        return g

    def switch_connection_high_k_cluster(self, g, index, del_index, n, neigs):
        """
        @param g: graph to work on
        @param index: id of main node
        @param del_index: id of node to disconnect
        @param n: number of nodes in g
        @param neigs: ids of neighbors of index
        @return: graph g after switch
        """
        g.delete_edges((index, del_index))
        switch_group = []
        new_neig = None
        for node in neigs:
            if node != del_index:
                switch_group.extend(g.neighbors(node))
        switch_group = list(set(switch_group) - set(neigs) - {index})
        if switch_group:
            if len(switch_group) == 1:
                new_neig = switch_group[0]
            else:
                cumulative_degree_plus_a = np.cumsum((np.array(g.vs(switch_group).degree()) + self.a) ** 2.0)
                new_neig = switch_group[np.searchsorted(cumulative_degree_plus_a,
                                                        random.randint(1, cumulative_degree_plus_a[-1]))]
        else:
            while 1:
                new_neig = random.randint(0, n - 1)
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
            print(i)
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
            if i % 100000 == 0:
                if g.is_static():
                    return g
        return g

    def basic_algorithm_watch_graph(self, g, T):
        """Copy of basic algorithm which counts number of switches
        from the beginning of simulation and computes other things.
        Separated from basic_algorithm() to keep speed of that function.
        @param g: graph to work on
        @param T: number of time steps
        @return (graph, dict): g after applying algorithm,
        and dictionary of some parameters of graph
        """
        save_step = int(T) / 10000 or 1
        n = len(g.vs())
        res = {'time': [], 'switches_sum': [], 'domains': [], 'components': [], 'degree': []}
        switches_sum = 0
        for i in range(T):
            # get one node and randomly select one of it's neighbors
            index = int(rand()*n)
            neigs = g.neighbors(index)
            if not neigs:
                continue
            neig_index = random.choice(neigs)
            # compare attributes of two nodes
            vertex_attrs = g.vs(index)["f"][0]
            neighbor_attrs = g.vs(neig_index)["f"][0]
            m = np.count_nonzero((vertex_attrs == neighbor_attrs))
            # decide what to do according to common attributes
            if m == 0:
                self.switch_function(g, index, neig_index, n, neigs)
                switches_sum += 1
            elif m != self.f and rand() < m*1.0/self.f:
                change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
                vertex_attrs[change_attr] = neighbor_attrs[change_attr]
            # save some data
            if i % save_step == 0:
                res['time'].append(i)
                res['switches_sum'].append(switches_sum)
                # res['components'].append(g.get_components())
                # res['domains'].append(g.get_domains())
                # res['degree'].append(list((x, y) for x, _, y in g.degree_distribution().bins()))  # remember about g.degree()
        res['time'].append(T)
        res['switches_sum'].append(switches_sum)
        return res, g

    def func_star(self, chain):
        """Converts `f([1,2])` to `f(1,2)` call.
        It's necessary when using Pool.map_async() function"""
        return self.basic_algorithm_count_switches(*chain)

    def get_average_component_for_q(self, N, q, T, av_over):
        """This method calls base_algorithm for av_over times
        and computes average largest component and domain.
        @param N: number of nodes in graph
        @param q: number of possible values of node's attributes
        @param T: number of time steps for base_algorithm
        @param av_over: number of base_algorithm executions
        @return float *4: average largest component and domain and their std
        """
        biggest_clusters = []
        biggest_domain = []
        for i in range(av_over):
            g = AxGraph.random_graph_with_attrs(N, self.av_k, self.f, q)
            g = self.basic_algorithm(g, T)
            biggest_clusters.append(g.get_largest_component() * 1.0 / N)
            biggest_domain.append(g.get_largest_domain() * 1.0 / N)
        return np.mean(biggest_clusters), np.mean(biggest_domain), np.std(biggest_clusters), np.std(biggest_domain)

    def get_average_component_for_q_multi(self, N, q, T, av_over):
        """This method calls base_algorithm for av_over times
        and computes average largest component and domain. It uses multiprocessing
        for spawning child processes.
        @param N: number of nodes in graph
        @param q: number of possible values of node's attributes
        @param T: number of time steps for base_algorithm
        @param av_over: number of base_algorithm executions
        @return float *4: average largest component and domain and their std
        """
        biggest_clusters = []
        biggest_domain = []
        k = {}
        dist = {}
        clustering = {'global': 0.0, 'av_local_nan': 0.0, 'av_local_zero': 0.0}
        path = []
        numbers_sd = {'comps': 0.0, 'doms': 0.0}
        
        def append_result(res_g):
            """This function is called from the main process
            to append results to lists.
            @param res_g: object of AxGraph class
            """
            biggest_clusters.append(res_g.get_largest_component() * 1.0 / N)
            biggest_domain.append(res_g.get_largest_domain() * 1.0 / N)
            for x, _, y in res_g.degree_distribution().bins():
                if x in k:
                    k[x] += y * 1.0 / (1.0 * av_over)
                else:
                    k[x] = y * 1.0 / (1.0 * av_over)
            clustering['global'] += res_g.transitivity_undirected() / (1.0 * av_over)
            clustering['av_local_nan'] += res_g.transitivity_avglocal_undirected(mode="nan") / (1.0 * av_over)
            clustering['av_local_zero'] += res_g.transitivity_avglocal_undirected(mode="zero") / (1.0 * av_over)
            for comp in res_g.get_components().values():
                if comp in dist:
                    dist[comp] += 1.0 / (1.0 * av_over)
                else:
                    dist[comp] = 1.0 / (1.0 * av_over)
            path.append(res_g.average_path_length())
            numbers_sd['comps'] += res_g.get_number_of_components() * 1.0 / av_over
            numbers_sd['doms'] += res_g.get_number_of_domains_properly() * 1.0 / av_over
            return True

        pool = Pool(processes=self.processes)
        for i in range(av_over):
            g = AxGraph.random_graph_with_attrs(N, self.av_k, self.f, q)
            pool.apply_async(basic_algorithm_multi, args=(self.mode, self.f, g, T), callback=append_result)
        pool.close()
        pool.join()
        pool.terminate()
        res = (np.mean(biggest_clusters), np.mean(biggest_domain), np.std(biggest_clusters), np.std(biggest_domain))

        write_object_to_file(res, 'q='+str(q)+'.data')
        write_object_to_file(k, self.mode + '_degree_N' + str(N) + '_q' + str(q) + '_av' + str(av_over) + '.data')
        write_object_to_file(dist, self.mode + '_components_N' + str(N) + '_q' + str(q) + '_av' + str(av_over) + '.data')
        write_object_to_file(clustering, self.mode + '_clustering_N' + str(N) + '_q' + str(q) + '_av' + str(av_over) + '.data')
        write_object_to_file({'path': 1.0 * sum(path) / (len(path) * 1.0)}, self.mode + '_N' + str(N) + '_path_q' + str(q) + '_av' + str(av_over) + '.data')
        write_object_to_file(numbers_sd, self.mode + '_q' + str(q) + '_N' + str(N) + '_numbers_sd.data')
        return res

    def return_many_graphs_multi(self, N, q, T, g_number):
        """This method calls base_algorithm for g_number times
        and returns that number of final graphs.
        @param N: number of nodes in graph
        @param q: number of possible values of node's attributes
        @param T: number of time steps for base_algorithm
        @param g_number: number of base_algorithm executions
        @return list: list of graphs
        """
        graphs = []

        def append_result(res_g):
            """This function is called from the main process
            to append results to lists.
            @param res_g: object of AxGraph class
            """
            graphs.append(res_g)
            return True
        pool = Pool(processes=self.processes)
        for i in range(g_number):
            g = AxGraph.random_graph_with_attrs(N, self.av_k, self.f, q)
            pool.apply_async(basic_algorithm_multi, args=(self.mode, self.f, g, T), callback=append_result)
        pool.close()
        pool.join()
        pool.terminate()
        return graphs
    
    def get_data_for_qsd(self, N, T, av_over_q, q_list):
        """Method with loop over q to get data for plots.
        @param N: number of nodes in graph
        @param T: number of time steps in base algorithm
        @param av_over_q: number of repetitions for one q
        @param q_list: list of q's values to iterate over
        @return dict: dictionary with lists of q, components and domains and their std
        """
        q_list.sort()
        res = {'q': [], 's': [], 'd': [], 's_std': [], 'd_std': []}
        for q in q_list:
            start_time = time.time()
            comp, dom, comp_std, dom_std = self.compute_comp_dom(N, q, T, av_over_q)
            res['q'].append(q)
            res['s'].append(comp)
            res['d'].append(dom)
            res['s_std'].append(comp_std)
            res['d_std'].append(dom_std)
            log.info("computing components and domains %s times for q = %s finished in %s minutes"\
                     % (av_over_q, q, round((time.time()-start_time)/60.0, 2)))
            self.try_sleep()
        return res

    def save_data(self, N, T, av_over_q, q_list):
        """Method with loop over q to get data for plots.
        @param N: number of nodes in graph
        @param T: number of time steps in base algorithm
        @param av_over_q: number of repetitions for one q
        @param q_list: list of q's values to iterate over
        @return dict: dictionary with lists of q, components and domains and their std
        """
        q_list.sort()
        for q in q_list:
            start_time = time.time()
            self.compute_comp_dom(N, q, T, av_over_q)
            log.info("computing components and domains %s times for q = %s finished in %s minutes" \
                     % (av_over_q, q, round((time.time() - start_time) / 60.0, 2)))
            self.try_sleep()
        return

    def compute_time(self, N, q, T, av_over, term):
        """This method calls find_times_multi for av_over times
        and computes time of termalization of component and domain. It uses multiprocessing
        for spawning child processes.
        @param N: number of nodes in graph
        @param q: number of possible values of node's attributes
        @param T: max number of time steps for base_algorithm
        @param av_over: number of find_times_multi executions
        @return float *4: average time for component and domain and their std
        """
        time_cluster = []
        time_domain = []
        time_all = []

        def append_result(res_times):
            """This function is called from the main process
            to append results to lists.
            @param res_g: object of AxGraph class
            """
            time_cluster.append(res_times['component'])
            time_domain.append(res_times['domain'])
            time_all.append(res_times['all'])
            return True
        pool = Pool(processes=self.processes)
        for i in range(av_over):
            g = AxGraph.random_graph_with_attrs(N, self.av_k, self.f, q)
            pool.apply_async(find_times_multi_properly, args=(self.mode, self.f, g, T, term), callback=append_result)
        pool.close()
        pool.join()
        pool.terminate()
        res = (np.mean(time_cluster), np.mean(time_domain), np.mean(time_all), np.std(time_cluster),
               np.std(time_domain), np.std(time_all))
        write_object_to_file(res, str(self.mode) + '_time_q='+str(q)+'.data')
        return res

    def get_times_for_qsd(self, N, T, av_over, q_list, term):
        """Method with loop over q to get times of termalization of s and d.
        @param N: number of nodes in graph
        @param T: number of time steps in base algorithm
        @param av_over_q: number of repetitions for one q
        @param q_list: list of q's values to iterate over
        @return dict: dictionary with lists of q, components and domains and their std
        """
        q_list.sort()
        res = {'q': [], 's': [], 'd': [], 'a': [], 's_std': [], 'd_std': [], 'a_std': []}
        for q in q_list:
            start_time = time.time()
            comp_time, dom_time, all_time, comp_std, dom_std, all_std = self.compute_time(N, q, T, av_over, term)
            res['q'].append(q)
            res['s'].append(comp_time)
            res['d'].append(dom_time)
            res['a'].append(all_time)
            res['s_std'].append(comp_std)
            res['d_std'].append(dom_std)
            res['a_std'].append(all_std)
            log.info("computing times for S and D %s times for N = %s, q = %s finished in %s min"\
                     % (av_over, N, q, round((time.time()-start_time)/60.0, 2)))
            self.try_sleep()
        return res

    def watch_many_graphs(self, N, T, q_list):
        """Method is calling base_algorithm_watch_graph in loop
        for many values of q and saves results into dictionaries.
        @param N: number of nodes in graph
        @param T: number of time steps in base algorithm
        @param q_list: list of values of q to iterate
        @result (dict, dict): dictionaries with final graphs
        and with all data for them
        """
        result = {}
        for q in q_list:
            start_time = time.time()
            g = AxGraph.random_graph_with_attrs(N, self.av_k, self.f, q)
            result[q] = self.basic_algorithm_watch_graph(g, T)
            log.info("watching graph algorithm for N = %s, T = %s, q = %s finished in %s minutes"\
                     % (N, T, q, round((time.time()-start_time)/60.0, 2)))
            self.try_sleep()
        return result


def get_distribution(values):
    res = {}
    for value in values:
        if value in res:
            res[value] += 1
        else:
            res[value] = 1
    return [int(x) for x, y in res.items()], [y for x, y in res.items()]


if __name__ == '__main__':
    # simulation = AxSimulation('normal', 4.0, 3, 4, [])
    # q_list = [2, 50, 100, 500]
    # simulation.watch_many_graphs(500, 1000000, q_list)

    model = 'k_plus_a'
    for q in [5000, 5000, 5000, 5000, 5000]:
        simulation = AxSimulation(model, 4.0, 3, 4, [])
        g = AxGraph.random_graph_with_attrs(N=500, q=q)
        res, g = simulation.basic_algorithm_watch_graph(g, 1000000)  # type: AxGraph
        print(q)
        print(res['time'][-1], res['switches_sum'][-1], 100.0 * res['switches_sum'][-1] / res['time'][-1])
        print()

        # print(read_object_from_file('q=2.data'))
        # break
        # __a = 1
        # simulation = AxSimulation(model, 4.0, 3, 4, [])
        # simulation.save_data(500, 2000000, 2, [2])
        # g = simulation.basic_algorithm(g, 2000000)  # type: AxGraph
        # print(model, q, g.is_static())
        # # g.pickle('/home/tomaszraducha/Pulpit/graph_{}_q{}.data'.format(model, q))
        # ig.plot(g)
