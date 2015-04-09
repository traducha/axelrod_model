#-*- coding: utf-8 -*-

import time
from base import *
import random
import numpy as np
from numpy.random import random as rand
from matplotlib import pyplot as plt




times = 5000
#### Checking first time of execution###

start_time = time.time()
for i in range(times):
    ############################
    
    g = AxGraph.random_graph_with_attrs(500, 4, 3, 30)
    x = g.get_edgelist()
    
    ############################
total1 = time.time() - start_time



############### Checking second time of execution ############

start_time = time.time()
for i in range(times):
    #############################
    
    g = AxGraph.random_graph_with_attrs(500, 4, 3, 30)
    x = g.es()
    
    #############################
total2 = time.time() - start_time


############### Checking time of execution #############
end_time = time.time() - start_time
print("\n\n--- first executed in %s seconds ---" % (total1))
print("\n\n--- second executed in %s seconds ---" % (total2))
########################################################
