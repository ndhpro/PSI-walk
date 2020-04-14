import os
import sys
from main import run_file


def load_keys(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [str(line)[:-1] for line in lines]


def run_folder(path, keys):
    for _, _, files in os.walk(path):
        for file in files:
            # Check analysis status
            flag = True
            for _, _, rfiles in os.walk('results/'):
                if file in rfiles:
                    flag = False

            if flag:
                if path.endswith('/'):
                    file_path = sys.argv[1] + file
                else:
                    file_path = sys.argv[1] + '/' + file
                run_file(file_path, keys)
    return 0


if __name__ == "__main__":
    # Create_report folder
    if not os.path.exists('results/'):
        os.makedirs('results/')

    keys = load_keys('key.txt')
    path = sys.argv[1]
    if os.path.isdir(path):
        run_folder(path, keys)
    elif os.path.isfile(path):
        run_file(path, keys)
    else:
        print('Input is not a directory or normal file')
