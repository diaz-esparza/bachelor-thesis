from os.path import join as jo
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import numpy as np
import os
import os.path as osp
import pennylane as qml

FILE = osp.dirname(__file__)
ROOT = jo(FILE, "..", "..")
DATAPATH = jo(ROOT, "storage", "outputs", "datasets", "embeddings", "quantum")


## Acceso a datos
def mini_setup():
    X_train = []
    y_train = []
    for i in os.scandir(jo(DATAPATH, "train")):
        if i.name.endswith("npy"):
            X_train.append(np.load(i.path))
            label_path = osp.splitext(i.path)[0] + ".txt"
            with open(label_path, "rt") as f:
                y_train.append(int(f.read()))
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []

    for i in os.scandir(jo(DATAPATH, "valid")):
        if i.name.endswith("npy"):
            X_test.append(np.load(i.path))
            label_path = osp.splitext(i.path)[0] + ".txt"
            with open(label_path, "rt") as f:
                y_test.append(int(f.read()))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def preprocessing(
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_value: float = np.pi,
):
    scaler = MinMaxScaler(feature_range=(0, max_value))
    return scaler.fit_transform(X_train), scaler.transform(X_test)


## QSVM
N_QBITS = 16
DEV = qml.device("default.qubit", wires=N_QBITS)


@qml.qnode(DEV)
def kernel(x1, x2):
    qml.templates.AngleEmbedding(x1, wires=range(N_QBITS))
    qml.adjoint(qml.templates.AngleEmbedding(x2, wires=range(N_QBITS)))
    return qml.probs()


def kernel_matrix(A, B):
    return np.array([[kernel(a, b)[0] for b in B] for a in A])


def main():
    X_train_raw, y_train, X_test_raw, y_test = mini_setup()
    X_train, X_test = preprocessing(X_train_raw, X_test_raw)

    ## Clasificación
    qsvm = SVC(kernel=kernel_matrix, verbose=True).fit(X_train, y_train)  ##type:ignore
    predictions = qsvm.predict(X_test)
    print("QSVM:", accuracy_score(predictions, y_test))

    X_train, X_test = preprocessing(X_train_raw, X_test_raw, max_value=1)
    svm = SVC().fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print("Classical SVM:", accuracy_score(predictions, y_test))

    X_train, X_test = preprocessing(X_train_raw, X_test_raw, max_value=1)
    mlp = MLPClassifier(hidden_layer_sizes=32).fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    print("MLP:", accuracy_score(predictions, y_test))


if __name__ == "__main__":
    main()
