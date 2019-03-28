#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from matplotlib import pyplot as plt
import time
from base_a import *
import numpy as np
import math
import igraph as ig
import os

N = 50
f = 3
av_k = 4.0
mode = 'cluster'


def plot_cd(T, comp, dom):
    plt.plot(range(T), comp, 'b')
    plt.plot(range(T), dom, 'r')
    plt.show()


def get_layout(g, niter=None, seed=None):
    return g.layout_graphopt(niter=niter, seed=seed, node_charge=0.0001, node_mass=50, spring_length=1)


def plot_g(g, count, layout, niter=None):
    global count_plots
    count_plots += 1
    ig.plot(g, 'plots/{}.png'.format(count), layout=layout, bbox=(720, 720))
    return get_layout(g, niter=niter, seed=layout)


sim = AxSimulation(mode, av_k, f, 1, [])
init_states = rand((N, f))
global count_plots
count_plots = 1

##############################################################################################################
# faza pierwsza
q = 2
T = 10000

init_g = AxGraph.random_graph_with_attrs(N, sim.av_k, sim.f, q, edge_width=2, init_states=init_states)
g = init_g.copy(q, init_states)
g.set_all_colors()
layout = get_layout(g, niter=500)
init_layout = layout
ig.plot(g, 'plots/0.png', layout=layout, bbox=(720, 720))
for j in xrange(50):
    layout = plot_g(g, count_plots, layout, niter=5)

comp, dom = [], []
count_changed = 0
for i in xrange(T):
    g, changed = sim.basic_algorithm(g, 1, N)
    dom.append(g.get_largest_domain())
    comp.append(g.get_largest_component())
    if changed:
        if count_changed % 10 == 0:
            print i
            for j in xrange(5):
                layout = plot_g(g, count_plots, layout, niter=5)
        count_changed += 1

for j in xrange(50):
    layout = plot_g(g, count_plots, layout, niter=5)
plot_cd(T, comp, dom)

to_double = count_plots
for i in range(to_double):
    os.system('cp plots/{}.png plots/{}.png'.format(i, to_double + (to_double - i - 1)))
    count_plots += 1

##############################################################################################################
# faza druga
q = 20
T = 10000

g = init_g.copy(q, init_states)
g.set_all_colors()
layout = init_layout

first_plot = count_plots
for j in xrange(50):
    layout = plot_g(g, count_plots, layout, niter=5)

comp, dom = [], []
count_changed = 0
for i in xrange(T):
    g, changed = sim.basic_algorithm(g, 1, N)
    dom.append(g.get_largest_domain())
    comp.append(g.get_largest_component())
    if changed:
        if count_changed % 10 == 0:
            print i
            for j in xrange(5):
                layout = plot_g(g, count_plots, layout, niter=5)
        count_changed += 1

for j in xrange(100):
    layout = plot_g(g, count_plots, layout, niter=5)
plot_cd(T, comp, dom)

# to_double = count_plots - first_plot
old_count_plot = count_plots
for i in range(first_plot, old_count_plot):
    os.system('cp plots/{}.png plots/{}.png'.format(i, old_count_plot + (old_count_plot - i - 1)))
    count_plots += 1

##############################################################################################################
# faza trzecia
q = 1000
T = 3000

g = init_g.copy(q, init_states)
g.set_all_colors()
layout = init_layout

first_plot = count_plots
for j in xrange(50):
    layout = plot_g(g, count_plots, layout, niter=5)

comp, dom = [], []
count_changed = 0
for i in xrange(T):
    g, changed = sim.basic_algorithm(g, 1, N)
    dom.append(g.get_largest_domain())
    comp.append(g.get_largest_component())
    if changed:
        if count_changed % 10 == 0:
            print i
            for j in xrange(5):
                layout = plot_g(g, count_plots, layout, niter=5)
        count_changed += 1

for j in xrange(50):
    layout = plot_g(g, count_plots, layout, niter=5)
plot_cd(T, comp, dom)

# to_double = count_plots - first_plot
old_count_plot = count_plots
for i in range(first_plot, old_count_plot):
    os.system('cp plots/{}.png plots/{}.png'.format(i, old_count_plot + (old_count_plot - i - 1)))
    count_plots += 1

##############################################################################################################

os.system('ffmpeg -r 30 -pattern_type sequence -s 720x720 -start_number 0 -i "plots/%d.png" -q:v 1 atest.mp4')
os.system('rm plots/*.png')
