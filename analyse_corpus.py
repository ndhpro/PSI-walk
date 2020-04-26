import os
import sys
from pathlib import Path
import networkx as nx
import csv

data = list()

# with open(Path('corpus/malware.txt'), 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         obj = dict()
#         obj['len'] = len(line.split(' '))
#         data.append(obj)

with open(Path('corpus/benign.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        obj = dict()
        obj['len'] = len(line.split(' '))
        data.append(obj)

csv_header = ['len']

try:
    with open('psi_walk_stat.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        for d in data:
            writer.writerow(d)
except IOError:
    print("I/O error")



