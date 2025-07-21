## UNUSED FOR NOW
from os.path import join as jo
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import keras
import numpy as np
import os
import os.path as osp
import pennylane as qml

from silence_tensorflow import silence_tensorflow

silence_tensorflow()


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


## Definición en pennylane
N_QBITS = 16
N_LAYERS = 2
DEV = qml.device("lightning.qubit", wires=N_QBITS)


@qml.qnode(DEV)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QBITS))
    qml.templates.BasicEntanglerLayers(weights, wires=range(N_QBITS))
    state_0 = [[1], [0]]
    y = state_0 * np.conj(state_0).T
    return [qml.expval(qml.Hermitian(y, wires=[0]))]  ##type:ignore


def main():
    X_train_raw, y_train, X_test_raw, y_test = mini_setup()
    X_train, X_test = preprocessing(X_train_raw, X_test_raw)

    ## Exportación a keras
    weight_shapes = {"weights": (N_LAYERS, N_QBITS)}
    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=N_QBITS)

    model = keras.models.Sequential([qlayer])
    opt = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics="accuracy")
    ## Early stopping
    ea_cb = keras.callbacks.EarlyStopping(
        monitor="accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
    )
    ## Entrenamiento
    model.fit(X_train, y_train, epochs=100, callbacks=[ea_cb])
    print("QNN", model.evaluate(X_test, y_test))


if __name__ == "__main__":
    main()
