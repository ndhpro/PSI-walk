import numpy as np
import networkx as nx
from copy import copy, deepcopy


class Environment():
    def __init__(self, graph, root, keys):
        self.graph = graph
        self.init_state = dict({'node': root, 'edge': dict()})
        self.gone_edge = set()
        self.keys = keys

    def reset(self):
        self.gone_edge = set()
        return self.init_state

    def get_avail_action(self, state):
        cur = state['node']
        ret = list()

        for nei in self.graph.adj[cur]:
            if nei != cur:
                ret.append(nei)

        return ret

    def step(self, action, state):
        freq = 5 * self.graph.out_degree(action) / (len(self.graph.edges()))
        reward_ = 1 / (1 + np.exp(-freq))
        reward = reward_
        tmp = str(action).lower()
        for key in self.keys:
            k, theta = key.split()
            if tmp == k:
                reward = int(theta) * (1+reward_)

        cur = state['node']
        edge = deepcopy(state['edge'])
        if (cur, action) in self.gone_edge:
            edge[str(cur) + ' ' + str(action)
                 ] = edge.get(str(cur) + ' ' + str(action), 0) + 1
            edge = dict(sorted(edge.items()))
            reward //= (2**(edge[str(cur) + ' ' + str(action)]))

        if (action, action) in self.graph.edges():
            reward *= 2

        self.gone_edge.add((cur, action))
        next_state = dict({'node': action, 'edge': edge})

        done = True
        if len(self.get_avail_action(next_state)):
            done = False

        return next_state, reward, done
