"""
evaluate.py

Återanvändbar utvärderings-funktion som kapslar in utvärderingslogiken från
model/notebooks/01_cnn_training.ipynb (Sektion 8).

Kör modellen mot testsetet (data som modellen aldrig sett, varken under
träning eller validering) och visualiserar resultatet med en confusion
matrix, precis som i notebooken.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# CIFAR-10-klassnamn på svenska, i exakt samma ordning som one-hot-kodningen
# i data_loader.py (samma ordning som CLASS_NAMES i backend/routers/predict.py,
# fast där på engelska eftersom det är webb-API:et).
CLASS_NAMES_SV = [
    "flygplan",
    "bil",
    "fågel",
    "katt",
    "hjort",
    "hund",
    "groda",
    "häst",
    "båt",
    "lastbil",
]


def evaluate_model(model, x_test, y_test, show_plot=True):
    """
    Utvärderar den tränade modellen mot testsetet och visar en confusion
    matrix över resultatet.

    Steg:
    1. model.evaluate() ger den övergripande test_loss och test_accuracy -
       ett enda mått på hur bra modellen presterar totalt sett på data
       den aldrig sett förut.
    2. model.predict() ger sannolikheter för varje klass, för varje testbild.
       Vi tar fram den klass med högst sannolikhet (argmax) som modellens
       faktiska prediktion.
    3. confusion_matrix() jämför de predikterade klasserna mot de sanna
       klasserna, och bygger en 10x10-matris som visar exakt vilka klasser
       modellen blandar ihop (t.ex. katt/hund), inte bara att den har fel.
    4. Matrisen visualiseras som en heatmap med seaborn, med svenska
       klassnamn på båda axlarna.

    Args:
        model (tf.keras.Model): den tränade modellen som ska utvärderas.
        x_test, y_test: testdata och tillhörande one-hot-labels.
        show_plot (bool): om True, visas confusion matrix-plotten direkt
                           (via plt.show()). Sätt till False om du vill
                           bygga vidare på figuren själv innan den visas.

    Returns:
        tuple: (test_loss, test_accuracy, cm)
            test_loss (float): modellens loss på testsetet.
            test_accuracy (float): modellens accuracy på testsetet.
            cm (np.ndarray): 10x10 confusion matrix (rader = sann klass,
                kolumner = predikterad klass).
    """
    # 1. Övergripande loss och accuracy på testsetet
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # 2. Prediktioner för alla testbilder, omvandlade till klassindex
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)  # y_test är one-hot, gör om till index

    # 3. Bygg confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # 4. Visualisera som heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES_SV,
        yticklabels=CLASS_NAMES_SV,
    )
    plt.xlabel("Predikterad klass")
    plt.ylabel("Sann klass")
    plt.title(f"Confusion Matrix (test-accuracy: {test_accuracy * 100:.2f}%)")

    if show_plot:
        plt.show()

    return test_loss, test_accuracy, cm