# -*- coding: utf-8 -*-

import pprint
import glob
import re
import time
import logging as log
from matplotlib import pyplot as plt
import numpy as np
import base
from base import *


PROCESSES = 8
AV_OVER = 400
T = 4000000
q_list = [2, 5, 50, 100]
mode_list = ['normal', 'BA', 'cluster', 'k_plus_a', 'k_plus_a2', 'high_k_cluster']
N_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def get_domains_number(mode, q, N):
    sim = base.AxSimulation(mode, 4.0, 3, PROCESSES, [])
    base.__a = 1
    sim.set_a(1)

    graphs = sim.return_many_graphs_multi(N, q, T, AV_OVER)
    domains = []
    components = []
    for g in graphs:
        domains.append(g.get_number_of_domains_properly())
        components.append(g.get_number_of_components())

    dom_av = np.mean(domains)
    dom_std = np.std(domains)
    com_av = np.mean(components)
    com_std = np.std(components)

    res = {
        'q': q,
        'N': N,
        'mode': mode,
        'av_over': AV_OVER,
        'dom_av': dom_av,
        'dom_std': dom_std,
        'com_av': com_av,
        'com_std': com_std,
    }
    base.write_object_to_file(res, '{}_domains_number_N{}_q{}_av{}.data'.format(mode, N, q, AV_OVER))


def run_in_loop():
    for q in q_list:
        for mode in mode_list:
            for N in N_list:
                start_time = time.time()
                get_domains_number(mode, q, N)
                log.info("computation of domains for q {}, mode {}, N {} finished in {} min"
                         .format(q, mode, N, round((time.time() - start_time) / 60.0, 2)))


def fetch_results():
    res = {}
    pattern = re.compile(r'([a-zA-Z2_]*)_domains_number_N([0-9]{1,4})_q([0-9]{1,4})_av[0-9]{1,4}\.data')
    for _file in glob.glob("*.data"):
        print(_file)
        match = pattern.match(_file)
        mode, N, q = match.groups()
        N = int(N)

        key = ''.join([mode, '_q=', q])
        if key not in res:
            res[key] = {}

        results = base.read_object_from_file(_file)
        res[key][N] = {
            'dom_av': results['dom_av'],
            'dom_std': results['dom_std'],
            'com_av': results['com_av'],
            'com_std': results['com_std'],
        }
    return res


def plot_results(res, show=False):
    for q_mode, results in res.items():
        dom_av = []
        dom_std = []
        com_av = []
        com_std = []
        n_list = []
        for N, values in results.items():
            dom_av.append(values['dom_av'])
            dom_std.append(values['dom_std'])
            com_av.append(values['com_av'])
            com_std.append(values['com_std'])
            n_list.append(N)

        plt.errorbar(n_list, com_av, yerr=com_std, fmt='o', fillstyle='none', ecolor='blue')
        plt.errorbar(n_list, dom_av, yerr=dom_std, fmt='o', fillstyle='none', ecolor='red')
        plt.title(q_mode)
        plt.xlabel('N')
        plt.ylabel('number of doms (red), comps (blue)')

        if show:
            plt.show()
        else:
            plt.savefig(''.join([q_mode, '.png']), format='png')
        plt.clf()


if __name__ == '__main__':
    # run_in_loop()
    res = fetch_results()
    pprint.pprint(res)
    plot_results(res)
