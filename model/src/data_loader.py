"""
data_loader.py

Återanvändbar modul för att ladda, förbehandla och dela upp CIFAR-10-datasetet
i tränings-, validerings- och testset.

Läser INTE via tf.keras.datasets.cifar10.load_data() (se README/notebook för
varför — Kerasgrimman interna cache-/hash-hantering visade sig opålitlig).
Istället läses de redan nedladdade pickle-filerna direkt från disk, med samma
logik som togs fram och testades i model/notebooks/01_cnn_training.ipynb.

Filformatet är dokumenterat på https://www.cs.toronto.edu/~kriz/cifar.html:
varje batch-fil är en "picklad" Python-dict med nycklarna b'data'
(10000 bilder x 3072 värden: 1024 röda + 1024 gröna + 1024 blåa pixlar)
och b'labels' (10000 heltalsetiketter, 0-9).

Förbehandlingen som görs här:
- Normalisering av pixelvärden från [0, 255] till [0.0, 1.0] (float32)
- One-hot encoding av labels (för categorical_crossentropy)
- Stratifierad uppdelning av träningsdatan i tränings- och valideringsset
"""

import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Sequential

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


def load_cifar10(val_size=0.1, random_state=42):
    """
    Laddar in hela CIFAR-10-datasetet, förbehandlar det och delar upp det i
    tränings-, validerings- och testset.

    Steg:
    1. Läser in de 5 träningsbatcharna (data_batch_1-5) och slår ihop dem.
    2. Läser in testbatchen (test_batch).
    3. Normaliserar pixelvärden från [0, 255] till [0.0, 1.0] (float32),
       vilket gör att nätverket tränar snabbare och mer stabilt.
    4. One-hot-encodar labels (t.ex. 3 -> [0,0,0,1,0,0,0,0,0,0]), eftersom
       vår modell kommer använda categorical_crossentropy som loss-funktion.
    5. Delar upp träningsdatan i ett tränings- och ett valideringsset
       (stratifierat, så klassfördelningen förblir balanserad i båda).

    Parametrar:
        val_size (float): andel av träningsdatan som avsätts till validering
                           (default 0.1 = 10%, dvs 5000 av 50000 bilder).
        random_state (int): frö för slumpgeneratorn, för reproducerbara resultat.

    Returnerar:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
        Alla bilder har formen (N, 32, 32, 3) med värden i [0.0, 1.0].
        Alla labels har formen (N, 10) (one-hot).
    """
    # 1. Läs in och slå ihop de 5 träningsbatcharna
    train_batches = [
        load_batch(os.path.join(CIFAR_DIR, f"data_batch_{i}")) for i in range(1, 6)
    ]
    x_train_full = np.concatenate([images for images, _ in train_batches], axis=0)
    y_train_full = np.concatenate([labels for _, labels in train_batches], axis=0)

    # 2. Läs in testbatchen
    x_test, y_test = load_batch(os.path.join(CIFAR_DIR, "test_batch"))

    # 3. Normalisera pixelvärden till [0.0, 1.0]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 4. One-hot-encoda labels (10 klasser: 0-9)
    y_train_full = to_categorical(y_train_full, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # 5. Dela upp träningsdatan i tränings- och valideringsset, stratifierat
    #    på klass så att fördelningen förblir balanserad i båda delarna.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_augmentation_layer():
    """
    Bygger ett Keras-lager som utför enkel data augmentation på bilder:
    slumpmässig horisontell spegling, liten rotation och liten zoom.

    Augmentation ska ENDAST appliceras på träningsdatan (aldrig på
    validerings- eller testdatan), och endast under träning - lagret
    är inaktivt vid inferens/prediktion (t.ex. i webbdemot).

    Varför just dessa transformationer:
    - RandomFlip("horizontal"): CIFAR-10-objekt (bilar, djur, flygplan osv.)
      ser lika rimliga ut spegelvända, så detta ger gratis variation utan
      att förvränga bilden.
    - RandomRotation(0.05): roterar bilden max ca ±18 grader, vilket ger
      lite variation utan att förstöra små detaljer i de redan lågupplösta
      32x32-bilderna.
    - RandomZoom(0.1): simulerar att objektet är lite närmare/längre bort
      eller lite förskjutet i bilden.

    Returnerar:
        tf.keras.Sequential: ett lager som kan läggas till direkt i en
        modell, eller appliceras separat på en dataset-pipeline.
    """
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")