import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

# Loading corpus
corpus = list()
labels = list()
with open(Path('corpus/malware.txt'), 'r') as f:
    lines = f.readlines()
corpus.extend([data[:-1] for data in lines])
labels.extend([0] * len(lines))

with open(Path('corpus/benign.txt'), 'r') as f:
    lines = f.readlines()
corpus.extend([data[:-1] for data in lines])
labels.extend([1] * len(lines))

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(corpus)
X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen=100)
vocab_size = len(tokenizer.word_index) + 1
print(f'Vocab size: {vocab_size}, Unique nodes:', end=' ')
unique_word = 0
for k in tokenizer.word_docs:
    if tokenizer.word_docs[k] == 1:
        unique_word += 1
print(unique_word)

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=16, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(units=16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.15, random_state=42)
print('Train on', X_train.shape, ', test on', X_test.shape)

batch_size = 512
history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size,
          verbose=2, validation_split=3/17)

print()
y_pred = model.predict(X_test, verbose=1, batch_size=batch_size)
y_pred = [y >= 0.5 for y in y_pred]

print(metrics.classification_report(y_test, y_pred, digits=4))
print(metrics.confusion_matrix(y_test, y_pred))
