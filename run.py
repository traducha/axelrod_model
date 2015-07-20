#-*- coding: utf-8 -*-

import time
from matplotlib import pyplot as plt
import logging as log
from multiprocessing import Pool
import sys
import os
import base
from base import *

def loop_over_q():
    """This function makes several things. Goal is to visualize
    some behavior of graphs in simulation. Function uses multiprocessing
    to run 4 processes at once. It plots switches in time, writes graphs to file
    and writes clusters vs. q to file.
    """
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    N = 500
    Sim = AxSimulation(mode=2, av_k=4.0, f=3)
    clusters = {'q': [], 's': []}
    times = 20000000
    q_list = [3] + [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))][::3] + [1300, 9000]
    q_list = [(q_list[i], q_list[i+1], q_list[i+2], q_list[i+3]) for i in range(0, len(q_list), 4)]
    for q in q_list:
        start_time = time.time()
        g1 = AxGraph.random_graph_with_attrs(N, Sim.av_k, Sim.f, q[0])
        g2 = AxGraph.random_graph_with_attrs(N, Sim.av_k, Sim.f, q[1])
        g3 = AxGraph.random_graph_with_attrs(N, Sim.av_k, Sim.f, q[2])
        g4 = AxGraph.random_graph_with_attrs(N, Sim.av_k, Sim.f, q[3])
        pool_agrs = [[g1, times], [g2, times], [g3, times], [g4, times]]
        
        pool = Pool(processes=4)
        res = pool.map_async(Sim.func_star, pool_agrs)
        pool.close()
        pool.join()
        pool.terminate()
        result = res.get()
        
        for j in range(4):
            g, x, y = result[j]
            log.info("algorithm for q = %s executed in %s seconds" % (q[j], round((time.time() - start_time), 4)))
            
            g.write_pickle('OUT/graph_N='+str(N)+'_q='+str(q[j])+'_T='+str(times))
            clusters['s'].append(len(g.clusters()[0]) * 1.0 / N)
            clusters['q'].append(q[j])
            
            plt.plot(x[::1000], y[::1000])
            plt.title("Network with N = %s nodes, f = %s, q = %s" % (N, Sim.f, q[j]))
            plt.xlabel('time step')
            plt.ylabel('total number of switches')
            plt.savefig("OUT/switches_N="+str(N)+"_q="+str(q[j])+".png", format="png")
            plt.clf()
        log.info("%s percent of algorithm executed" % round((100.0 * (q_list.index(q) + 1.0) / len(q_list)), 1) )
        
    write_clusters_to_file(clusters, name='OUT/clusters.txt')
    return True

def plot_sd_vs_q(name):
    """Use in interacive mode to plot results.
    @param name: name of file with data
    """
    r = read_object_from_file(name)
    plt.scatter(r['q'], r['s'], color='blue')
    plt.scatter(r['q'], r['d'], color='red')
    plt.xlim([1, 10000])
    plt.ylim([0, 1])
    plt.xscale('log')
    plt.show()
    return True

def watch_one_graph(g, T):
    """This function runs simulation for one graph
    and saves largest component and domain for every time step.
    @param g: graph to start with
    @param T: number of time steps
    @return: dictionary with lists to plot
    """
    N = len(g.vs())
    Sim = AxSimulation(2, 4.0, 3)
    res = {'t': [], 's': [], 'd': []}
    for t in range(T):
        g = Sim.basic_algorithm(g, 1)
        if t % 100000 == 0:
            res['t'].append(t)
            res['s'].append(g.get_largest_component() * 1.0 / N)
            res['d'].append(g.get_largest_domain() * 1.0 / N)
            print t, g.is_switch_possible()
    write_object_to_file(res, 'play_in_time_N='+str(N)+'.data')
    plt.plot(res['t'], res['s'], color='blue')
    #plt.plot(res['t'], res['d'], color='red')
    plt.title("Largest component and domain in time, N = %s" % N)
    plt.xlabel('time step')
    plt.ylabel('largest component/domain')
    plt.savefig('play_in_time_N='+str(N)+'.png', format="png")
    return res

def main(N=500, av_q=20, T=1000000):
    log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=log.INFO)
    # read initial arguments
    if '-p' in sys.argv:
        processes = int(sys.argv[sys.argv.index('-p')+1])
        log.info("%s child processes will be spawn" % processes)
    else:
        raise Exception("Use switch '-p' to define number of processes")
    
    if '-m' in sys.argv:
        mode = sys.argv[sys.argv.index('-m')+1]
        log.info("mode of simulation is: %s" % mode)
    else:
        raise Exception("Use switch '-m' to define mode of simulation")

    if '-a' in sys.argv:
        a = sys.argv[sys.argv.index('-a')+1]
        log.info("constant 'a' is: %s" % a)
    elif mode in ['k_plus_a', '4', 4]:
        raise Exception("Use switch '-a' to define constant 'a' for simulation mode 'k_plus_a'")
    
    if '--rest' in sys.argv:
        rest = sys.argv[sys.argv.index('--rest')+1]
        rest = [float(i) for i in rest.split('-')]
        log.info("rest mode is: %s" % rest)
    else:
        rest = []
    # set simulation parameters
    q_list = [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))] #51 points in log scale
    simulation = AxSimulation(mode, 4.0, 3, processes, rest)
    simulation.set_a(a)
    base.__a = a
    # run simulation and save results
    main_time = time.time()
    res = simulation.get_data_for_qsd(N, T, av_q, q_list)
    write_object_to_file(res, 'OUT/res_N='+str(N)+'_q_times_'+str(av_q)+'_mode='+mode+'.data')
    log.info("main function executed in %s minutes" % round((time.time()-main_time)/60.0, 2))
    return

def watch_graphs():
    q_list = [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))] #51 points in log scale
    l = [129, 152, 177, 208, 243, 284, 333, 389, 456]
    q_list += [l[i]+(l[i+1]-l[i])/3 for i in range(len(l)-1)]\
                + [l[i]+2*(l[i+1]-l[i])/3 for i in range(len(l)-1)]
    q_list.sort()
    simulation = AxSimulation('normal', 4.0, 3, 4, [])
    for q in q_list[::12]:
        res = simulation.watch_many_graphs(500, 1000000, [q])
        write_object_to_file(res, 'OUT/q='+str(q)+'_watch.data')
    return


def get_distribution(values):
    res = {}
    for value in values:
        if value in res:
            res[value] += 1
        else:
            res[value] = 1
    return [int(x) for x, y in res.items()], [y for x, y in res.items()]

def plot_graphs():
    q_list = [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))] #51 points in log scale
    l = [129, 152, 177, 208, 243, 284, 333, 389, 456]
    q_list += [l[i]+(l[i+1]-l[i])/3 for i in range(len(l)-1)]\
                + [l[i]+2*(l[i+1]-l[i])/3 for i in range(len(l)-1)]
    q_list.sort()
    N = 500.0
    for q in q_list[::12]:
        res = read_object_from_file('OUT/q='+str(q)+'_watch.data')[q]
        print 'wczytało dane dla q = ' + str(q)
        os.mkdir('/home/tomaszraducha/PycharmProjects/Networks/PLOTS/'+str(q))
        os.mkdir('/home/tomaszraducha/PycharmProjects/Networks/PLOTS/'+str(q)+'/distributions')

        plt.plot(res['time'], res['switches_sum'])
        plt.xlabel('time step')
        plt.ylabel('number of switches')
        plt.savefig('PLOTS/'+str(q)+'/q='+str(q)+'switches.png', format="png")
        plt.clf()

        comp = [max(i.values())/N for i in res['components']]
        plt.plot(res['time'], comp)
        plt.xlabel('time step')
        plt.ylabel('largest component')
        plt.savefig('PLOTS/'+str(q)+'/q='+str(q)+'largest_comp.png', format="png")
        plt.clf()

        dom = [max(i.values())/N for i in res['domains']]
        plt.plot(res['time'], dom)
        plt.xlabel('time step')
        plt.ylabel('largest domain')
        plt.savefig('PLOTS/'+str(q)+'/q='+str(q)+'largest_dom.png', format="png")
        plt.clf()

        comps = [len(i) for i in res['components']]
        plt.plot(res['time'], comps)
        plt.xlabel('time step')
        plt.ylabel('number of components')
        plt.savefig('PLOTS/'+str(q)+'/q='+str(q)+'number_of_comps.png', format="png")
        plt.clf()

        doms = [len(i) for i in res['domains']]
        plt.plot(res['time'], doms)
        plt.xlabel('time step')
        plt.ylabel('number of domains')
        plt.savefig('PLOTS/'+str(q)+'/q='+str(q)+'number_of_doms.png', format="png")
        plt.clf()

        plt.plot(res['time'], comps, color='blue')
        plt.plot(res['time'], doms, color='red')
        plt.xlabel('time step')
        plt.ylabel('# of components (blue), domains (red)')
        plt.savefig('PLOTS/'+str(q)+'/q='+str(q)+'number_of_comps_and_doms.png', format="png")
        plt.clf()

        for i, t in enumerate(res['time'][::200]):
            os.mkdir('/home/tomaszraducha/PycharmProjects/Networks/PLOTS/'+str(q)+'/distributions/'+str(t))

            plt.scatter([j[0] for j in res['degree'][i]], [j[1] for j in res['degree'][i]])
            plt.xlabel('degree of node')
            plt.ylabel('number of nodes')
            plt.savefig('PLOTS/'+str(q)+'/distributions/'+str(t)+'/t='+str(t)+'q='+str(q)+'degree.png', format="png")
            plt.clf()

            x, y = get_distribution(res['components'][i].values())
            plt.scatter(x, y)
            plt.xlabel('size of component')
            plt.ylabel('number of components')
            plt.savefig('PLOTS/'+str(q)+'/distributions/'+str(t)+'/t='+str(t)+'q='+str(q)+'component_dist.png', format="png")
            plt.clf()

            x, y = get_distribution(res['domains'][i].values())
            plt.scatter(x, y)
            plt.xlabel('size of domain')
            plt.ylabel('number of domains')
            plt.savefig('PLOTS/'+str(q)+'/distributions/'+str(t)+'/t='+str(t)+'q='+str(q)+'domain_dist.png', format="png")
            plt.clf()

            print 'narysowało ' + str(i) + '  /  ' + str(len(res['time'][::200]))

        print 'narysowało dla q = ' + str(q)
    return

def check_sd_vs_n(q, N_list):
    res = {}
    av_q = 4
    simulation = AxSimulation('BA', 4.0, 3, 4, [])
    for n in N_list:
        main_time = time.time()
        res[n] = simulation.get_data_for_qsd(n, 500000, av_q, [q])
        log.info("N = %s executed in %s minutes" % (n, round((time.time()-main_time)/60.0, 2)))
    write_object_to_file(res, 'sd_vs_N_q='+str(q)+'_q_times_'+str(av_q)+'_mode='+'BA'+'.data')

if __name__ == "__main__":
    # q = 5
    # N = [10, 20, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # check_sd_vs_n(q, N)

    q_list = [int(1.17**i) for i in range(2,59) if int(1.17**i) != int(1.17**(i-1))] #71 points in log scale
    # l = [94, 111, 129, 152, 177, 208, 243, 284, 333, 389, 456]
    # q_list += [l[i]+(l[i+1]-l[i])/3 for i in range(len(l)-1)]\
    #             + [l[i]+2*(l[i+1]-l[i])/3 for i in range(len(l)-1)]
    # q_list = q_list[::3]
    q_list.sort()
    s =[]
    d =[]
    s_ = []
    d_ = []
    Q = []
    for q in q_list:
        try:
            x, y, z, w = read_object_from_file('cluster500test/q=%s.data' % q)
        except:
            continue
        s.append(x)
        d.append(y)
        s_.append(z)
        d_.append(w)
        Q.append(q)

    res = read_object_from_file('res_N=500_q_times_400_mode=cluster.data')

    plt.scatter(res['q'], res['s'], color='blue')
    plt.scatter(res['q'], res['d'], color='red')
    plt.scatter(Q, s, color='black')
    plt.scatter(Q, d, color='black')
    plt.xlim([1, 10000])
    plt.ylim([0, 1])
    plt.xscale('log')
    # plt.yscale('log')
    plt.show()

    plt.scatter(q_list, s_, color='blue')
    plt.scatter(q_list, d_, color='red')
    plt.xlim([1, 10000])
    plt.xscale('log')
    plt.show()

    s1500 =[]
    d1500 =[]
    s_1500 = []
    d_1500 = []
    for q in q_list:
        x, y, z, w = read_object_from_file('N1500/q=%s.data' % q)
        s1500.append(x)
        d1500.append(y)
        s_1500.append(z)
        d_1500.append(w)

    s1000 =[]
    d1000 =[]
    s_1000 = []
    d_1000 = []
    for q in q_list:
        x, y, z, w = read_object_from_file('N1000/q=%s.data' % q)
        s1000.append(x)
        d1000.append(y)
        s_1000.append(z)
        d_1000.append(w)

    s_prim = []
    d_prim = []
    for i, q in enumerate(q_list[:-1]):
        s_prim.append((s[i+1]-s[i])/(q_list[i+1]-q_list[i]))
        d_prim.append((d[i+1]-d[i])/(q_list[i+1]-q_list[i]))

    s_prim2 = []
    d_prim2 = []
    for i, q in enumerate(q_list[1:-1]):
        s_prim2.append((s_prim[i]-s_prim[i-1])/(q_list[i]-q_list[i-1]))
        d_prim2.append((d_prim[i]-d_prim[i-1])/(q_list[i]-q_list[i-1]))

    # plt.scatter(q_list[1:-1], s_prim2, color='blue')
    # plt.scatter(q_list[1:-1], d_prim2, color='red')

    plt.scatter(q_list, s1500, color='blue')
    plt.scatter(q_list, d1500, color='red')

    # plt.scatter(q_list, s_, color='blue')
    # plt.scatter(q_list, s_1000, color='red')
    # plt.scatter(q_list, s_1500, color='green')

    plt.xlim([1, 10000])
    plt.ylim([0, 1])
    plt.xscale('log')
    # plt.yscale('log')
    plt.show()

    s = s

    #watch_graphs()
    #plot_graphs()
    #main(N=500, av_q=4, T=200000)
    #watch_graphs()

    #loop_over_q()
#     g = read_graph_from_file('OUT/graph_N=500_q=243_T=20000000')
#     print g.is_static()
#     print g.is_switch_possible()
#     watch_one_graph(g, 10000000)
    
#     N = 500
#     av_q = 100
#     T = 1200000






# rysowanie
# a = plt.plot(r['q'], r['s'], color='blue')
# b = plt.scatter(r['q'], r['d'], color='red')
# plt.legend([a, b], ['Largest component', 'Largest domain'])
# plt.ylabel('fraction of # of nodes')
# plt.xlabel('q')
# plt.title('Network with 500 nodes, # of attrs f = 3.\nResults averaged over 400 realizations.')
# plt.plot(r['q'], r['s'], color='blue')
# plt.xlim([1, 10000])
# plt.ylim([0, 1])
# plt.xscale('log')
# plt.show()
# plt.clf()

# wsp klastrowania globalny g.transitivity_undirected()
# lokalny g.transitivity_avglocal_undirected(mode="nan") albomode="zero"
