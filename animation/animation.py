#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from matplotlib import pyplot as plt
from base_a import *
import igraph as ig
import os


mode = 'high_k_cluster'
speed_times = 3
Q = (2, 20, 1000)
T = (10000, 10000, 2000)
N = 50
f = 3
av_k = 4.0


def plot_cd(t, comp, dom):
    plt.plot(range(t), comp, 'b')
    plt.plot(range(t), dom, 'r')
    plt.show()


def get_layout(g, niter=None, seed=None):
    return g.layout_graphopt(niter=niter, seed=seed, node_charge=0.0001, node_mass=50, spring_length=1)


def plot_g(g, count_plots, layout, niter=None):
    ig.plot(g, 'plots/{}.png'.format(count_plots), layout=layout, bbox=(720, 720))
    return get_layout(g, niter=niter, seed=layout), count_plots + 1


def animate(q=None, T=None, init_g=None, layout=None, init_states=None, sim=None, count_plots=1, speed_times=1):
    g = init_g.copy(q, init_states)
    g.set_all_colors()

    first_plot = count_plots
    for j in xrange(50):
        layout, count_plots = plot_g(g, count_plots, layout, niter=5)

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
                    layout, count_plots = plot_g(g, count_plots, layout, niter=5)
            count_changed += 1

    for j in xrange(50):
        layout, count_plots = plot_g(g, count_plots, layout, niter=5)
    plot_cd(T, comp, dom)

    old_count_plot = count_plots
    for i in range(first_plot, old_count_plot):
        if i % speed_times == 0:
            os.system('cp plots/{}.png plots/{}.png'.format(i, old_count_plot + (old_count_plot - i - 1) / speed_times))
            count_plots += 1
    return count_plots


sim = AxSimulation(mode, av_k, f, 1, [])
init_states = rand((N, f))
init_g = AxGraph.random_graph_with_attrs(N, sim.av_k, sim.f, q=2, edge_width=2, init_states=init_states)
count_plots = 1
kwargs = {
    'init_g': init_g,
    'layout': get_layout(init_g, niter=500),
    'init_states': init_states,
    'sim': sim,
    'speed_times': speed_times,
}

count_plots = animate(q=Q[0], T=T[0], count_plots=count_plots, **kwargs)
count_plots = animate(q=Q[1], T=T[1], count_plots=count_plots, **kwargs)
count_plots = animate(q=Q[2], T=T[2], count_plots=count_plots, **kwargs)

os.system('ffmpeg -r 30 -pattern_type sequence -s 720x720 -start_number 0 -i "plots/%d.png" -q:v 1 {}{}.mp4'.format(mode, speed_times))
os.system('rm plots/*.png')
