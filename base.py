#-*- coding: utf-8 -*-

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
    
    # TODO True jeżeli wszystki możliwe niepołączone pary maja 0 wspólnych atr, a wszystkie połączone maja wszystkie takie same attr
    def is_dynamically_trapped(self):
        """This method finds out if graph is in dynamic equilibrium,
        i.e. there is no chance for interaction between two
        nodes ever, only edges keep switching.
        @param g: graph
        @return boolean: True if graph is trapped
        """
        N = len(self.es())
        for i in range(N):
            for j in range(i+1, N):
                if any(self.vs(i)["f"][0] == self.vs(j)["f"][0]):
                    return False
        return True

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
        uniq = {}
        for i, attrs in enumerate(self.vs()["f"]):
            for key, value in uniq.items():
                if all(attrs == value):
                    domains[key] += 1
                    break
            else:
                uniq[i] = attrs
                domains[i] = 1
        return domains

    def get_largest_domain(self):
        """Returns number of nodes in largest domain of graph.
        Its easy to change this function to return also number of domains.
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
    
    def get_number_of_domains(self):
        """Returns number of domains in graph.
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
        return len(domains)
    
    def get_largest_domain_and_number(self):
        """Returns number of nodes in largest domain of graph
        and number of domains. It's faster then using get_largest_domain
        and get_number_of_domains separately.
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
        return max(domains.values()), len(domains)
    
#########################################################################
# Functions used in running simulation on several processes.            #
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

SWITCH_MAP = {'1': switch_connection_while, '2': switch_connection_BA,
              1: switch_connection_while, 2: switch_connection_BA,
              'normal': switch_connection_while, 'BA': switch_connection_BA}

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
    return g

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
    
    def __init__(self, mode, av_k, f, processes, rest):
        """Set up parameters for simulation.
        @param mode: mode of simulation, switching behavior depends on it
        @param av_k: average degree of nodes
        @param f: number of attributes (every attribute can have q different values)
        @param processes: number of parallel processes to spawn
        """
        self.switch_connection_while = switch_connection_while
        
        if mode in ['normal', '1', 1]:
            self.switch_function = self.switch_connection_while
        elif mode in ['BA', '2', 2]:
            self.switch_function = self.switch_connection_BA
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

    def basic_algorithm_watch_graph(self, g, T):
        """Copy of basic algorithm which counts number of switches
        from the beginning of simulation and computes other things.
        Separated from basic_algorithm() to keep speed of that function.
        @param g: graph to work on
        @param T: number of time steps
        @return (graph, list, list): g after applying algorithm,
        list with number of time steps, list with sum of switches from the beginning
        """
        save_step = int(T) / 10000 or 1
        n = len(g.vs())
        res = {'time': [], 'switches_sum': [], 'domains': [], 'components': [], 'degree_dist': []}
        switches_sum = 0
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
                switches_sum += 1
            elif m != self.f and rand() < m*1.0/self.f:
                change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
                vertex_attrs[change_attr] = neighbor_attrs[change_attr]
            #save some data
            if i % save_step == 0:
                res['time'].append(i)
                res['switches_sum'].append(switches_sum)
                res['components'].append(g.clusters())
                res['domains'].append(g.get_domains())
                res['degree_dist'].append(g.degree_distribution())
        return g, res

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
        
        def append_result(res_g):
            """This function is called from the main process
            to append results to lists.
            @param res_g: object of AxGraph class
            """
            biggest_clusters.append(res_g.get_largest_component() * 1.0 / N)
            biggest_domain.append(res_g.get_largest_domain() * 1.0 / N)
            return True
        pool = Pool(processes=self.processes)
        for i in range(av_over):
            g = AxGraph.random_graph_with_attrs(N, self.av_k, self.f, q)
            pool.apply_async(basic_algorithm_multi, args=(self.mode, self.f, g, T), callback=append_result)
        pool.close()
        pool.join()
        pool.terminate()
        return np.mean(biggest_clusters), np.mean(biggest_domain), np.std(biggest_clusters), np.std(biggest_domain)
    
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

if __name__ == '__main__':
    basic_algorithm_multi('1', 3, 3, 3)
    x = AxGraph.random_graph_with_attrs(500, 4, 3, 30)
    print x
    print type(x)