import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Preparing data
root_path = Path('data')
folders = ['bashlite/', 'mirai/', 'others/', 'benign/']
corpus = list()
labels = list()


for folder in folders:
    for _, _, files in os.walk(root_path/folder):
        for name in files:
            try:
                with open(root_path/folder/name, 'r') as f:
                    doc = f.read().replace('\n', ' ')[:-1]
                corpus.append(doc)
                if 'benign' in folder:
                    labels.append(1)
                else:
                    labels.append(0)
            except Exception as e:
                print(e)

corpus, index = np.unique(corpus, axis=0, return_index=True)
labels = np.array(labels)[index]

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(corpus)
X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen=100)
print(X.shape)
np.save('corpus', X)
np.save('labels', labels)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)