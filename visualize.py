import os
import sys

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph


def run(fpath):
    G = nx.MultiDiGraph()
    G = nx.read_gexf(fpath)

    plt.figure(figsize=(10, 7))
    # pos = nx.planar_layout(G)
    node_color = [6 + int(G.degree(n)) for n in G.nodes()]

    nx.draw_networkx(G, node_size=100, node_color=node_color,
                     with_labels=True, font_size=9, edge_color='.4', cmap=plt.cm.get_cmap('Reds'), vmin=0)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1]
    for _, _, files in os.walk(path):
        for fname in files:
            if fname.endswith('.gexf'):
                fpath = path + fname
                run(fpath)
