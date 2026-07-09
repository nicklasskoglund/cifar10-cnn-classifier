"""
model.py

Definierar CNN-arkitekturen som används för att klassificera CIFAR-10-bilder.

Arkitekturen består av 3 convolutional-block med ökande filterdjup
(32 -> 64 -> 128), där varje block kombinerar Conv2D, BatchNormalization,
MaxPooling och Dropout, följt av ett fullt anslutet klassificeringshuvud.

BatchNormalization används för snabbare och stabilare träning (särskilt
värdefullt vid CPU-träning), och Dropout används genomgående för att
motverka overfitting.

Om augment=True läggs data augmentation-lagren (från data_loader.py) till
som de första lagren i modellen. Dessa lager är, precis som Dropout,
endast aktiva under träning (model.fit()) - vid prediktion/utvärdering
(model.predict(), model.evaluate()) passerar bilderna igenom oförändrade.
"""

from tensorflow.keras import layers, models

from data_loader import build_augmentation_layer


def build_model(input_shape=(32, 32, 3), num_classes=10, augment=True):
    """
    Bygger och kompilerar CNN-modellen för CIFAR-10-klassificering.

    Args:
        input_shape (tuple): Formen på indatabilderna (höjd, bredd, kanaler).
        num_classes (int): Antal klasser i output.
        augment (bool): om True, läggs data augmentation-lagren till som de
                        första lagren i modellen (endast aktiva under träning).

    Returns:
        tf.keras.Model: Kompilerad Keras Sequential-modell, redo för träning.
    """
    model = models.Sequential(name="cifar10_baseline_cnn")

    # --- Input ---
    model.add(layers.Input(shape=input_shape))

    # --- Data augmentation (endast aktiv under träning) ---
    if augment:
        model.add(build_augmentation_layer())

    # --- Conv-block 1 ---
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # --- Conv-block 2 ---
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # --- Conv-block 3 ---
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # --- Klassificeringshuvud ---
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # --- Kompilering ---
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model