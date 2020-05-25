import os
import sys
import subprocess
import time
import pandas as pd
from main import run_file


def proc_file(path):
    p = subprocess.Popen('python3 controller/main.py ' + path, shell=True)
    p.wait()


if __name__ == "__main__":
    # Create_report folder
    if not os.path.exists('results/'):
        os.makedirs('results/')

    malware = pd.read_csv(sys.argv[1]+'malware_v2-data.csv')

    for line in malware['md5'].values[:600]:
        continue_fl = False
        for _, dirs, _ in os.walk('final_report'):
            for dir in dirs:
                if str(file_name) in dir:
                    print('Found final report!')
                    continue_fl = True

        if not continue_fl:
            run_file(sys.argv[1]+'psi_graph-malwre/'+line+'.txt')