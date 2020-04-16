import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


X = np.load('corpus.npy')
labels = np.load('labels.npy')
print('Corpus size: ', X.shape)
vocab_size = 12910

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=32, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42)


batch_size = 128
model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=2, validation_split=0.25)

score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
