import os
import sys
from pathlib import Path
from main import run_file


def run_folder(path):
    for _, _, files in os.walk(path):
        for file_ in files:
            # Check analysis status
            flag = True
            for _, _, result_files in os.walk('results/'):
                if file_ in result_files:
                    flag = False

            if flag:
                file_path = path / file_
                run_file(file_path)
    return 0


if __name__ == "__main__":
    # Create_report folder
    result_path = Path('results/')
    if not Path.exists(result_path):
        Path.mkdir(result_path)

    path = Path(sys.argv[1])
    if Path.is_dir(path):
        run_folder(path)
    elif Path.is_file(path):
        run_file(path)
    else:
        print('Input is not a directory or normal file')
