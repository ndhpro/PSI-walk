import numpy as np
import networkx as nx


class Environment():
    def __init__(self, graph, root, keys):
        self.graph = graph
        self.init_state = root
        self.keys = keys

    def reset(self):
        return self.init_state

    def get_avail_action(self, cur):
        ret = list()

        for nei in self.graph.adj[cur]:
            if nei != cur:
                ret.append(nei)

        return ret

    def step(self, action):
        reward_ = self.graph.degree(action) / (2*len(self.graph.edges()))
        reward = reward_
        for key in self.keys:
            if str(action).lower() == key:
                reward = 1000 * reward_

        next_state = action

        done = True
        if len(self.get_avail_action(next_state)):
            done = False

        return next_state, reward, done
