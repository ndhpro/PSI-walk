import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import itertools

# Loading corpus
corpus = list()
labels = list()
with open(Path('corpusv2/malware.txt'), 'r') as f:
    lines = f.readlines()
corpus.extend([data[:-1] for data in lines])
labels.extend([0] * len(lines))

with open(Path('corpusv2/benign.txt'), 'r') as f:
    lines = f.readlines()
corpus.extend([data[:-1] for data in lines])
labels.extend([1] * len(lines))
labels = np.array(labels)

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
                    output_dim=10, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(units=10, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.1, random_state=42)
print('Train on', X_train.shape, ', test on', X_test.shape)

batch_size = 512
history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size,
                    verbose=2, validation_split=1/9)

print()
y_pred = model.predict(X_test, verbose=1, batch_size=batch_size)
y_pred = [y >= 0.5 for y in y_pred]

model.save('output/model_v2.h5')

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
plt.savefig('output/training_history_v2.png')


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
plt.savefig('output/confusion_matrix_v2.png')

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
plt.savefig('output/roc_v2.png')

# test zeroday
zeroday = list()
for _, _, files in os.walk(Path('datav2/zeroday/')):
    for file_ in files:
        file_path = Path('datav2/zeroday/') / file_
        with open(file_path, 'r') as f:
            line = f.read()
            line = line.replace('\n', ' ').strip()
            zeroday.append(line)
X_0 = tokenizer.texts_to_sequences(zeroday)
X_0 = pad_sequences(X_0, maxlen=100)
y_pred = model.predict(X_0, verbose=1, batch_size=batch_size)
print(np.hstack([np.array(files).reshape(-1,1),y_pred]))
y_pred = [y >= 0.5 for y in y_pred]
y_true = [0] * len(y_pred)
print(metrics.classification_report(y_true, y_pred, digits=4))
print()
print(metrics.confusion_matrix(y_true, y_pred))