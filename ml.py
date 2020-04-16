import os
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


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

# Saving for analysis
with open('corpus_malware.txt', 'w') as f:
    print(len(corpus[labels==0]))
    for i in corpus[labels==0]:
        f.write(str(i) + '\n')

with open('corpus_benign.txt', 'w') as f:
    print(len(corpus[labels==1]))
    for i in corpus[labels==1]:
        f.write(str(i) + '\n')

X_train, X_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.3, random_state=42)

# Vectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
print(X_train.shape, X_test.shape)

# Feature selection
model = SelectFromModel(
    LinearSVC(penalty="l1", dual=False, random_state=42).fit(X_train, y_train), prefit=True)
X_train = model.transform(X_train).A
X_test = model.transform(X_test).A

np.savez('data_ml', X_train, X_test, y_train, y_test)

# Normalizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape)

# Training
names = ['Logistic Regression', 'SVM', 'Decision Tree',
         'kNN', 'Naive Bayes', 'Random Forest']
models = [
    LogisticRegression(random_state=42, n_jobs=-1, class_weight='balanced'),
    SVC(random_state=42, class_weight='balanced'),
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    KNeighborsClassifier(n_jobs=-1),
    GaussianNB(),
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
]
hyperparams = [
    {},
    {},
    {},
    {},
    {},
    {}
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

for name, model, hyper, color in zip(names, models, hyperparams, colors):
    clf = GridSearchCV(model, param_grid=hyper, cv=5, n_jobs=-1) 
    print(name)

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    print(metrics.classification_report(y_test, y_hat))
    print(metrics.confusion_matrix(y_test, y_hat))
    print()

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_hat)
    auc = metrics.roc_auc_score(y_test, y_hat)
    plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

    # y_hat = clf.predict(X_train)
    # print(metrics.classification_report(y_train, y_hat))
    # print(metrics.confusion_matrix(y_train, y_hat))
    # print()

# Drawing ROC curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()