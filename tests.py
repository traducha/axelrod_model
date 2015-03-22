#-*- coding: utf-8 -*-

import time
import igraph as ig
import random
import numpy as np
from numpy.random import random as rand
from matplotlib import pyplot as plt

av_k = 4.0
n = 1
#p = av_k / (n - 1.0)
q = 2
f = 3

times = 10000
#### Checking first time of execution###

total1 = 0.0
for i in range(times):
    start_time = time.time()
    ##########################
    x = random.choice((1,2,3,4,5,6,7,8,9,0))
    ############################
    total1 += time.time() - start_time

############### Checking second time of execution #############



total2 = 0.0
for i in range(times):
    start_time = time.time()
    ###########################
    y = random.choice([])
    #############################
    total2 += time.time() - start_time


############### Checking time of execution #############
end_time = time.time() - start_time
print("\n\n--- first executed in %s seconds ---" % (total1/times))
print("\n\n--- second executed in %s seconds ---" % (total2/times))
########################################################

print x, y