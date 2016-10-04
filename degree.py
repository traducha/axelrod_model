#!/usr/bin/python
#-*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import logging as log
import time
import base

log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
N = 500
T = 3000000
processes = 12
q_list = [2, 80, 150, 5000]
av_over = 400
modes = ['normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2']


def get_degree(q, mode, av_over):
    sim = base.AxSimulation(mode, 4.0, 3, processes, [])
    base.__a = 1
    sim.set_a(1)

    graphs = sim.return_many_graphs_multi(N, q, T, av_over)
    k = {}
    for g in graphs:
        for x, _, y in g.degree_distribution().bins():
            if x in k:
                k[x] += y * 1.0 / (1.0 * av_over)
            else:
                k[x] = y * 1.0 / (1.0 * av_over)
    base.write_object_to_file(k, mode + '_degree_N500_q' + str(q) + '_av' + str(av_over) + '.data')
    return


def degree_for_modes_and_q(q_list, modes, av_over):
    for mode in modes:
        for q in q_list:
            main_time = time.time()
            get_degree(q, mode, av_over)
            log.info("computation of degree for mode %s and q %s finished in %s min" %
                     (mode, q, round((time.time()-main_time)/60.0, 2)))
    return


def get_degree_dynamical(q, mode, av_over):
    sim = base.AxSimulation(mode, 4.0, 3, processes, [])
    base.__a = 1
    sim.set_a(1)

    g = base.AxGraph.random_graph_with_attrs(N, 4.0, 3, q)
    g = sim.basic_algorithm(g, 2000000)
    k = {}
    for i in range(av_over):
        g = sim.basic_algorithm(g, 10000)
        for x, _, y in g.degree_distribution().bins():
            if x in k:
                k[x] += y * 1.0 / (1.0 * av_over)
            else:
                k[x] = y * 1.0 / (1.0 * av_over)
    base.write_object_to_file(k, mode + '_degree_N500_q' + str(q) + '_av' + str(av_over) + '.data')
    return


def degree_for_modes_and_q_dynamical(q_list, modes, av_over):
    for mode in modes:
        for q in q_list:
            main_time = time.time()
            get_degree_dynamical(q, mode, av_over)
            log.info("computation of degree for mode %s and q %s finished in %s min" %
                     (mode, q, round((time.time()-main_time)/60.0, 2)))
    return


def plot_degree(k_list, q=None, mode=None, save_as=False):
    ymin = 100
    for k in k_list:
        ymin = min(ymin, min(k.values(), key=lambda x: x or 10000.0) / 2.0)

    # normal plot
    for i, k in enumerate(k_list):
        if i == 0:
            color = 'blue'
        elif i == 1:
            color = 'red'
        elif i == 2:
            color = 'green'
        if k.keys()[0] == 0:
            plt.scatter(k.keys()[1:], k.values()[1:], color=color)
        else:
            plt.scatter(k.keys(), k.values())
    if mode:
        plt.suptitle(mode, fontsize=18)
    if q:
        plt.title('q = ' + str(q))
    plt.xlabel('k')
    plt.ylabel('number of nodes')
    if save_as:
        plt.savefig(save_as + '.png')
    else:
        plt.show()
    plt.clf()

    # log-log plot
    for i, k in enumerate(k_list):
        if i == 0:
            color = 'blue'
        elif i == 1:
            color = 'red'
        elif i == 2:
            color = 'green'
        if k.keys()[0] == 0:
            plt.scatter(k.keys()[1:], k.values()[1:], color=color)
        else:
            plt.scatter(k.keys(), k.values())
    if mode:
        plt.suptitle(mode, fontsize=18)
    if q:
        plt.title('q = ' + str(q))
    plt.xlabel('k')
    plt.ylabel('number of nodes')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ymin=ymin)
    if save_as:
        plt.savefig(save_as + '_log-log.png')
    else:
        plt.show()
    plt.clf()

    # y axis in log scale
    for i, k in enumerate(k_list):
        if i == 0:
            color = 'blue'
        elif i == 1:
            color = 'red'
        elif i == 2:
            color = 'green'
        if k.keys()[0] == 0:
            plt.scatter(k.keys()[1:], k.values()[1:], color=color)
        else:
            plt.scatter(k.keys(), k.values())
    if mode:
        plt.suptitle(mode, fontsize=18)
    if q:
        plt.title('q = ' + str(q))
    plt.xlabel('k')
    plt.ylabel('number of nodes')
    plt.yscale('log')
    plt.ylim(ymin=ymin)
    if save_as:
        plt.savefig(save_as + '_y-log.png')
    else:
        plt.show()
    plt.clf()
    return


def plot_degree_from_file(file_name=None):
    if not file_name:
        for mode in modes:
            for q in q_list:
                f_name = 'degree/' + mode + '_degree_N' + str(500) + '_q' + str(q) + '_av' + str(av_over) + '.data'
                f_name2 = 'degree/' + mode + '_degree_N' + str(1000) + '_q' + str(q) + '_av' + str(av_over) + '.data'
                f_name3 = 'degree/' + mode + '_degree_N' + str(2000) + '_q' + str(q) + '_av' + str(av_over) + '.data'
                k1 = base.read_object_from_file(f_name)
                k2 = base.read_object_from_file(f_name2)
                k3 = base.read_object_from_file(f_name3)
                for k in [k1, k2, k3]:
                    norm = sum(k.values()) * 1.0
                    for key, value in k.items():
                        k[key] = value * 1.0 / norm
                plot_degree([k1, k2, k3], q=q, mode=mode, save_as='double/'+f_name2.replace('degree', '').replace('.data', '').replace('1000', '500-1000-2000'))
    else:
        k = base.read_object_from_file(file_name)
        plot_degree(k)
    return

if __name__ == '__main__':
    # degree_for_modes_and_q(q_list, modes, av_over)
    plot_degree_from_file()
    # degree_for_modes_and_q_dynamical([5000], modes, av_over)
