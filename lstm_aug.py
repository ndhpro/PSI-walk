import itertools
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading corpus
X_train, X_test = list(), list()
y_train, y_test = list(), list()
with open(Path('corpus/train_1_aug.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        X_train.append(line[:-3])
        y_train.append(int(line[-2]))

with open(Path('corpus/test_1_aug.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        X_test.append(line[:-3])
        y_test.append(int(line[-2]))

print(len(X_train), len(X_test))

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=100)
vocab_size = len(tokenizer.word_index) + 1
print(f'Vocab size: {vocab_size}, Unique nodes:', end=' ')
unique_word = 0
for k in tokenizer.word_docs:
    if tokenizer.word_docs[k] == 1:
        unique_word += 1
print(unique_word)

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=100)

y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=16, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(units=16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# print(model.summary())

batch_size = 512
mc = ModelCheckpoint('output/model.h5', save_best_only=True,
                     save_weights_only=True)
history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size,
                    verbose=2, validation_split=1/9,
                    callbacks=[mc])

model.load_weights('output/model.h5')
y_pred = model.predict(X_test, verbose=1, batch_size=batch_size)
y_pred = [y >= 0.5 for y in y_pred]

print(metrics.classification_report(y_test, y_pred, digits=4))
print()

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r',
           label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'],
           color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig('output/training_history.png')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(metrics.confusion_matrix(y_test, y_pred), classes=[0, 1])
plt.savefig('output/confusion_matrix.png')

# Drawing ROC curve
plt.figure()
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr, tpr, color='blue', label='AUC = %0.4f' % (auc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.savefig('output/roc.png')
