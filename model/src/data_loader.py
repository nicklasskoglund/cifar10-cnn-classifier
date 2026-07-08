"""
data_loader.py

Återanvändbar modul för att ladda, förbehandla och dela upp CIFAR-10-datasetet.

Läser INTE via tf.keras.datasets.cifar10.load_data() (se README/notebook för
varför — Kerasgrimman interna cache-/hash-hantering visade sig opålitlig).
Istället läses de redan nedladdade pickle-filerna direkt från disk, med samma
logik som togs fram och testades i model/notebooks/01_cnn_training.ipynb.

Filformatet är dokumenterat på https://www.cs.toronto.edu/~kriz/cifar.html:
varje batch-fil är en "picklad" Python-dict med nycklarna b'data'
(10000 bilder x 3072 värden: 1024 röda + 1024 gröna + 1024 blåa pixlar)
och b'labels' (10000 heltalsetiketter, 0-9).
"""

import pickle
import numpy as np
import os

CIFAR_DIR = os.path.expanduser("~/.keras/datasets/cifar-10-batches-py")


def unpickle(file_path):
    """Läser en enskild CIFAR-10 batch-fil och returnerar dess dict."""
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_batch(file_path):
    """
    Läser en batch-fil och omformar bilddatan från den platta
    (N, 3072)-formen till (N, 32, 32, 3) - dvs N bilder, 32x32 pixlar, 3 färgkanaler.
    """
    batch = unpickle(file_path)
    images = batch[b"data"]
    labels = np.array(batch[b"labels"])

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return images, labels