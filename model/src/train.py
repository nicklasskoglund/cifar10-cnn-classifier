"""
train.py

Återanvändbar tränings-funktion som kapslar in träningslogiken från
model/notebooks/01_cnn_training.ipynb (Sektion 7).

Tanken är att samma träningslogik som redan är verifierad i notebooken ska
kunna köras även utanför Jupyter, t.ex. via run_pipeline.py i projektets
root. Detta är ett komplement till notebooken, inte en ersättning - notebooken
är fortfarande den plats där resultaten är dokumenterade och förklarade steg
för steg.
"""

import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_model

# Sökväg till där bästa modellvikterna sparas, relativt den här filen
# (model/src/train.py -> model/saved_models/best_model.keras), så det
# fungerar oavsett vilken working directory man kör ifrån.
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.keras")


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    input_shape=(32, 32, 3),
    num_classes=10,
    augment=True,
    epochs=30,
    batch_size=64,
    patience=5,
):
    """
    Bygger och tränar CNN-modellen, med samma callbacks och konfiguration
    som användes i notebookens Sektion 7.

    Callbacks som används:
    - EarlyStopping: avbryter träningen om val_loss inte förbättras under
      `patience` antal epoker i rad, och återställer automatiskt modellens
      vikter till den bästa epoken (restore_best_weights=True). Detta
      motverkar overfitting och sparar tid genom att inte köra alla 30
      epoker i onödan om modellen redan har slutat förbättras.
    - ModelCheckpoint: sparar modellens vikter till disk varje gång val_loss
      förbättras (save_best_only=True), så att den bästa versionen alltid
      finns kvar på disk - oavsett vad som händer resten av träningen
      (t.ex. om modellen börjar overfitta igen efter en bra epok).

    Args:
        x_train, y_train: träningsdata och tillhörande one-hot-labels.
        x_val, y_val: valideringsdata och tillhörande one-hot-labels.
        input_shape (tuple): formen på indatabilderna, skickas till build_model.
        num_classes (int): antal klasser, skickas till build_model.
        augment (bool): om data augmentation ska användas, skickas till build_model.
        epochs (int): max antal epoker att träna (EarlyStopping kan avbryta tidigare).
        batch_size (int): antal bilder per träningsbatch.
        patience (int): antal epoker utan förbättring innan EarlyStopping avbryter.

    Returns:
        tuple: (model, history)
            model (tf.keras.Model): den tränade modellen (bästa vikterna,
                tack vare restore_best_weights=True i EarlyStopping).
            history (tf.keras.callbacks.History): historik-objekt med
                loss/accuracy per epok, användbart för att plotta
                tränings- och valideringskurvor.
    """
    # Se till att mappen för sparade modeller finns innan träningen börjar,
    # annars kraschar ModelCheckpoint när den försöker spara första gången.
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        augment=augment,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return model, history