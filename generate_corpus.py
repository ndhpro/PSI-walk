import os
from pathlib import Path
import numpy as np
import random
from sklearn.model_selection import train_test_split


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
root_path = Path('data')
folders = ['bashlite/', 'mirai/', 'others/', 'benign/']
corpus = list()
labels = list()

for folder in folders:
    for _, _, files in os.walk(root_path/folder):
        for name in files:
            try:
                aug = 1
                with open(root_path/folder/name, 'r') as f:
                    doc = f.read().replace('\n', ' ')[:-1]
                words = doc.split(' ')
                enc = True
                for word in words:
                    if not 'sub' in word:
                        enc = False
                        break
                if enc:
                    continue
                corpus.append(doc)

                if 'benign' in folder:
                    labels.extend([0] * aug)
                else:
                    labels.extend([1] * aug)
            except Exception as e:
                print(e)

corpus, index = np.unique(corpus, axis=0, return_index=True)
labels = np.array(labels)[index]
X_train, X_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.3, random_state=2020)

X_aug, y_aug = [], []
for i in range(len(X_train)):
    aug = 1
    doc = X_train[i]
    # Data augmentation
    doc = doc.split(' ')
    while len(doc) > 4 and aug < 4:
        new_doc = random_augment(doc)
        if new_doc != doc:
            aug += 1
            X_aug.append(new_doc)
            y_aug.append(y_train[i])

X_train = list(X_train)
X_train.extend(X_aug)
y_train = list(y_train)
y_train.extend(y_aug)

# Saving for analysis
with open('corpus/train_1_aug.txt', 'w') as f:
    for i in range(len(X_train)):
        f.write(str(X_train[i]) + ' ' + str(y_train[i]) + '\n')

with open('corpus/test_1_aug.txt', 'w') as f:
    for i in range(len(X_test)):
        f.write(str(X_test[i]) + ' ' + str(y_test[i]) + '\n')

print(len(X_train), len(X_test))
