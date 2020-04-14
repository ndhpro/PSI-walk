import sys
import time
from copy import copy, deepcopy
import networkx as nx
import pandas as pd
from numpy.random import choice
from env import Environment
from agent import QLearningTable


def load_graph(path):
    G = nx.MultiDiGraph()
    G = nx.read_gexf(path)

    # Add super_root
    child = set()
    for u, v in G.edges():
        if u != v:
            child.add(v)
    start_edge = list()
    for n in G.nodes():
        if n not in child:
            start_edge.append(('ndhpro', n))
    G.add_edges_from(start_edge)

    print(len(G.nodes()), len(G.edges()))
    return G


def load_keys(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [str(line)[:-1] for line in lines]


def update(num_episode):
    global steps, all_costs

    for e in range(num_episode):
        i = 0
        cost = 0
        done = False
        state = env.reset()

        while not done:
            avail_action = env.get_avail_action(state)

            action = RL.choose_action(str(state), avail_action)
            state_, reward, done = env.step(action, state)

            cost += RL.learn(str(state), action,
                             reward, str(state_))

            state = deepcopy(state_)
            # print(state['node'], end=' ')

            i += 1
            if i == 50:
                break
        all_costs.append(cost)
        steps.append(i)
        # print()

        if (e+1) % (num_episode//10) == 0:
            RL.reduce_epsilon()


def get_final_path():
    t = time.time()
    Q = RL.get_q_table()
    # print('Length of full Q-table =', len(Q.index))
    # print('Full Q-table:')
    # print(Q)

    state = env.reset()
    i = 0
    done = 0
    print()
    while not done and i < 50:
        avail_actions = env.get_avail_action(state)
        try:
            state_action = Q.loc[str(state), avail_actions]
        except Exception as e:
            print('\n', e)
            break
        action = state_action.idxmax()
        state, _, done = env.step(action, state)
        print(state['node'], end=' ')
        if (state['node'], state['node']) in G.edges():
            print(state['node'], end=' ')
        i += 1
    print()
    print(i, round((time.time()-t), 2))

    RL.plot_results(steps, all_costs)


# Main
G = load_graph(sys.argv[1])
keys = load_keys('key.txt')
env = Environment(graph=G, root='ndhpro', keys=keys)
RL = QLearningTable(actions=G.nodes())
steps = []
all_costs = []

t = time.time()
update(min(1000, max(500, 2*len(G.edges()))))
print(round((time.time()-t), 2))

get_final_path()

