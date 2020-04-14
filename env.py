import numpy as np
import networkx as nx
import copy


class Environment():
    def __init__(self, graph, root, keys):
        self.graph = graph
        self.init_state = [root, dict()]
        self.gone_edge = set()
        self.keys = keys

    def reset(self):
        self.gone_edge = set()
        return self.init_state

    def get_avail_action(self, state):
        cur = state[0]
        ret = list()

        for nei in self.graph.adj[cur]:
            if nei != cur:
                ret.append(nei)

        return ret

    def step(self, action, state):
        reward_ = self.graph.degree(action) / (2*len(self.graph.edges()))
        reward = reward_
        for key in self.keys:
            k, theta = key.split()
            if str(action).lower() == k:
                reward = int(theta) * reward_

        cur, e = state
        edge = copy.deepcopy(e)
        if (cur, action) in self.gone_edge:
            edge[str(cur) + ' ' + str(action)] = edge.get(str(cur) + ' ' + str(action), 0) + 1
            edge = dict(sorted(edge.items()))
            reward //= (2**(edge[str(cur) + ' ' + str(action)]))

        self.gone_edge.add((cur, action))
        next_state = [action, edge]

        done = True
        if len(self.get_avail_action(next_state)):
            done = False

        return next_state, reward, done
