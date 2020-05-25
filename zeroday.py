import os
from pathlib import Path
import pandas as pd
import shutil


src = Path('datav2/malware/')
dst = Path('datav2/zeroday/')

data = pd.read_csv('data.csv')
for md5, label in data[['md5', 'label']].values:
    try:
        if int(label[-5:-1]) >= 2016 and 'Mirai' not in label:
            src_path = src / (md5 + '.txt')
            dst_path = dst / (md5 + label + '.txt')
            shutil.move(str(src_path), str(dst_path))
            print(md5, label)
    except Exception as e:
        print(e)
