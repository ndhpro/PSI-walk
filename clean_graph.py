import os
import sys
import json
import math
import pickle

import networkx as nx


def run(fpath):
    G = nx.MultiDiGraph()
    with open(fpath, 'r') as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if ' ' in line and line.find(' ') == line.rfind(' '):
            (u, v) = line.split(' ')
            G.add_edge(u, v)

    nx.write_gexf(G, fpath[fpath.find('\\')+1:].replace('.txt', '.gexf'))


if __name__ == "__main__":
    path = sys.argv[1]
    for _, _, files in os.walk(path):
        for fname in files:
            if fname.endswith('.txt'):
                fpath = path + fname
                run(fpath)
