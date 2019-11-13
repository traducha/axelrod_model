import igraph as ig
import time
from numpy.random import random as rand
import numpy as np
import csv


T = 100
N = 10
M = 20
J = 1.0
h = 0.0
B = 0.01


def initialize_graph(n, m):
    graph = ig.Graph.Erdos_Renyi(n=n, m=m)
    graph.vs()["spin"] = (np.int_(rand((N, 1)) * 2) * 2) - 1
    return graph


def energy_change(graph, spin_change, neigs):
    delta = 0.0
    for i in xrange(len(neigs)):
        delta += graph.vs(neigs[i])["spin"][0][0]
    delta = - (J * delta + h) * spin_change
    return delta


def new_vertex(n, exclude):
    new_v = np.int_(np.floor(rand() * n))
    if new_v in exclude:
        new_v = new_vertex(n, exclude)
    return new_v


def main_loop(graph, n, m, times):
    mag = []
    for i in xrange(times):
        v_index = np.int_(np.floor(rand() * n))
        e_index = np.int_(np.floor(rand() * m))

        # spin switching
        neigs = graph.neighbors(v_index)
        delta = energy_change(graph, -2 * graph.vs(v_index)["spin"][0][0], neigs)
        if delta <= 0.0 or rand() < np.exp(- B * delta):
            graph.vs(v_index)["spin"][0][0] *= -1

        # edge rewiring
        v_from, v_to = graph.get_edgelist()[e_index]
        v_to_keep, v_to_detach = (v_from, v_to) if rand() < 0.5 else (v_to, v_from)
        exclude = graph.neighbors(v_to_keep) + [v_to_keep, v_to_detach]
        v_to_new = new_vertex(n, exclude)
        delta = - J * graph.vs(v_to_keep)["spin"][0][0] * \
                (graph.vs(v_to_new)["spin"][0][0] - graph.vs(v_to_detach)["spin"][0][0])
        if delta <= 0.0 or rand() < np.exp(- B * delta):
            graph.delete_edges((v_from, v_to))
            graph.add_edges([(v_to_keep, v_to_new)])

        mag.append(np.sum(graph.vs()["spin"]))

    return graph, mag


if __name__ == '__main__':
    graph = initialize_graph(N, M)

    start = time.time()
    graph, mag = main_loop(graph, N, M, T)
    end = time.time()

    with open('magnetization2.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index'] + [i for i in xrange(len(mag))])
        writer.writerow(['value'] + mag)

    print("Main loop finished in {} s.".format(end - start))
