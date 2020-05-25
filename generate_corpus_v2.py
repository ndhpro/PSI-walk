import os
from pathlib import Path
import numpy as np
import random


def random_augment(words):
    new_words = []
    pre_word = ''
    for word in words:
        new_words.append(word)
        r = random.uniform(0, 3)
        if word == pre_word:
            new_words.extend([word]*int(r))
        pre_word = word

    new_words = ' '.join(new_words)

    return new_words


# Preparing data
root_path = Path('datav2')
folders = ['malware/', 'benign/']
corpus = list()
labels = list()

for folder in folders:
    for _, _, files in os.walk(root_path/folder):
        for name in files:
            try:
                aug = 1
                with open(root_path/folder/name, 'r') as f:
                    doc = f.read().replace('\n', ' ')[:-1]
                corpus.append(doc)

                # Data augmentation
                doc = doc.split(' ')
                if folder.startswith('malware'):
                    num_aug = 4
                else:
                    num_aug = 8
                while len(doc) > 4 and aug < num_aug:
                    new_doc = random_augment(doc)
                    if new_doc != doc:
                        aug += 1
                        corpus.append(new_doc)

                if 'benign' in folder:
                    labels.extend([1] * aug)
                else:
                    labels.extend([0] * aug)
            except Exception as e:
                print(e)

print(corpus[:10])
corpus, index = np.unique(corpus, axis=0, return_index=True)
labels = np.array(labels)[index]

# Saving for analysis
with open('corpusv2/malware.txt', 'w') as f:
    for i in corpus[labels == 0]:
        f.write(str(i) + '\n')

with open('corpusv2/benign.txt', 'w') as f:
    for i in corpus[labels == 1]:
        f.write(str(i) + '\n')