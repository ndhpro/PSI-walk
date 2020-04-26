import os
import sys
from pathlib import Path
import networkx as nx
import csv


classes = ['bashlite', 'mirai', 'others', 'benign']
csv_header = ['node', 'edge', 'label']
data = list()
path = Path(sys.argv[1])
for class_ in classes:
    for _, _, files in os.walk(path/class_):
        for file_ in files:
            fpath = path / class_ / file_
            G = nx.MultiDiGraph()
            try:
                with open(fpath, 'r') as f:
                    lines = f.readlines()
            except:
                continue

            for line in lines[2:]:
                line = line.strip()
                if ' ' in line and line.find(' ') == line.rfind(' '):
                    (u, v) = line.split(' ')
                    G.add_edge(u, v)

            obj = dict()
            obj['node'] = len(G.nodes())
            obj['edge'] = len(G.edges())
            if class_.startswith('benign'):
                obj['label'] = 'b'
            else:
                obj['label'] = 'r'
            data.append(obj)

try:
    with open('psi_graph_stat.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        for d in data:
            writer.writerow(d)
except IOError:
    print("I/O error")



