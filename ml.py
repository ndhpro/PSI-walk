from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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

# Drawing ROC curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
