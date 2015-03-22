#-*- coding: utf-8 -*-

import time
import igraph as ig
import random
import numpy as np
from numpy.random import random as rand
from matplotlib import pyplot as plt

#### Checking time of execution###
start_time = time.time()
##################################

#some initial parameters
av_k = 4.0
n = 500
p = av_k / (n - 1.0)
q = 2
f = 3
switches_sum = []
print("probability p: %s" % p)

#creating random graph and generating random attributes
g = ig.Graph.Erdos_Renyi(n, p)
print("initial average degree was: %s" % (np.sum(g.degree()) * 1.0 / n))
g.vs()["f"] = np.int_(rand((n, f))*q)

################# Main loop #####################
T = 1000
for i in range(T):
    #get one node and randomly select one of it's neighbors
    index = int(rand()*n)
    neigs = g.neighbors(index)
    if not neigs:
        switches_sum.append(np.sum(switches_sum))
        continue
    neig_index = random.choice(neigs)
    #compare attributes of two nodes
    vertex_attrs = g.vs(index)["f"][0]
    neighbor_attrs = g.vs(neig_index)["f"][0]
    m = np.count_nonzero((vertex_attrs == neighbor_attrs))
    #decide what to do according to common attributes
    if m == 0:
        g.delete_edges((index,neig_index))
        new_neig = random.choice(list(set(range(n)).difference(neigs)))
        g.add_edges([(index,new_neig)])
        switches_sum.append(np.sum(switches_sum)+1)
    elif m != f and rand() < m*1.0/f:
        change_attr = random.choice(np.where((vertex_attrs == neighbor_attrs) == False)[0])
        g.vs(index)["f"][0][change_attr] = g.vs(neig_index)["f"][0][change_attr]
        switches_sum.append(np.sum(switches_sum))
    
    
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

plt.plot(range(T), suma)
plt.show()



############### Checking time of execution #############
print("\n\n--- executed in %s seconds ---" % (time.time() - start_time))
########################################################