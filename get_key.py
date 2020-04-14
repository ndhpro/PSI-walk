import os
import sys
import networkx as nx
from main import load_graph


def get_key(path, nodes):
    try:
        G = load_graph(path)
        for v in G.nodes():
            if not (str(v).startswith('_') or str(v).startswith('sub_')):
                nodes.add(v)
    except:
        print(path)


if __name__ == "__main__":
    path = sys.argv[1]
    nodes = set()
    for _, _, files in os.walk(path):
        for fname in files:
            fname = path + '\\' + fname
            get_key(fname, nodes)
    nodes = list(nodes)
    nodes.sort()
    for v in nodes:
        print(v)
