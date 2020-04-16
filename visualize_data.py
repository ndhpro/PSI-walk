import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, Normalizer


def v2D(X_train, X_test, y_train, y_test):
    svd = TruncatedSVD(n_components=2)
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)

    # scaler = Normalizer()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    X_train_mal, X_train_beg = X_train[y_train==0], X_train[y_train==1]
    X_test_mal, X_test_beg = X_test[y_test==0], X_test[y_test==1]

    a, b = zip(*X_train_mal)
    plt.scatter(a, b, c='r', marker='.')
    a, b = zip(*X_train_beg)
    plt.scatter(a, b, c='b', marker='.')

    a, b = zip(*X_test_mal)
    plt.scatter(a, b, c='r', marker='.')
    a, b = zip(*X_test_beg)
    plt.scatter(a, b, c='b', marker='.')
    plt.show()


def v3D(X_train, X_test, y_train, y_test):
    svd = TruncatedSVD(n_components=2)
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)

    # scaler = Normalizer()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    X_train_mal, X_train_beg = X_train[y_train==0], X_train[y_train==1]
    X_test_mal, X_test_beg = X_test[y_test==0], X_test[y_test==1]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')

    a, b, c = zip(*X_train_beg)
    ax.scatter(a, b, c, c='b', label='Benign')
    a, b, c = zip(*X_train_mal)
    ax.scatter(a, b, c, c='r', label='Malware')

    a, b, c = zip(*X_test_mal)
    ax.scatter(a, b, c, c='r')
    a, b, c = zip(*X_test_beg)
    ax.scatter(a, b, c, c='b')

    ax.legend()
    plt.show()


if __name__ == "__main__":
    npzfile = np.load('data_ml.npz')
    X_train = npzfile['arr_0']
    X_test = npzfile['arr_1']
    y_train = npzfile['arr_2']
    y_test = npzfile['arr_3']
    print(X_train.shape, X_test.shape)
    v2D(X_train, X_test, y_train, y_test)