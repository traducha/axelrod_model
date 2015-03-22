#-*- coding: utf-8 -*-

import time
import igraph as ig
import random
import numpy as np
from numpy.random import random as rand
from matplotlib import pyplot as plt



times = 50
#### Checking first time of execution###

start_time = time.time()
for i in range(times):
    import base2
    ############################
total1 = time.time() - start_time

############### Checking second time of execution #############



start_time = time.time()
for i in range(times):
    import base
    #############################
total2 = time.time() - start_time


############### Checking time of execution #############
end_time = time.time() - start_time
print("\n\n--- first executed in %s seconds ---" % (total1))
print("\n\n--- second executed in %s seconds ---" % (total2))
########################################################
