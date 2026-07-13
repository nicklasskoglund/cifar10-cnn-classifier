"""
run_pipeline.py

CLI-ingång som kör hela CNN-pipelinen från kommandoraden, utan att behöva
öppna Jupyter: laddar CIFAR-10-datasetet, bygger modellen, tränar den och
utvärderar den mot testsetet - i ett enda kommando.

OBS: detta är ett ALTERNATIV till notebooken och webbgränssnittet, inte en
ersättning. Notebooken (model/notebooks/01_cnn_training.ipynb) är fortfarande
den plats där resultaten är dokumenterade och förklarade steg för steg, och
enstaka bild-inferens sker fortfarande bara via webb-UI:t (backend + frontend).
Den här filen är till för att snabbt kunna köra om hela tränings- och
utvärderingsprocessen från terminalen, t.ex. efter en kodändring.

Körs från projektets root med:
    python run_pipeline.py
"""

import os
import sys

import matplotlib.pyplot as plt

# model/src/ ligger inte i sys.path som standard, så vi lägger till den här
# innan vi importerar något därifrån. Detta gör det möjligt för moduler i
# model/src/ (data_loader.py, model.py, train.py, evaluate.py) att fortsätta
# använda sina enkla paket-interna imports (t.ex. "from model import build_model")
# oavsett att den här filen körs från root.
MODEL_SRC_DIR = os.path.join(os.path.dirname(__file__), "model", "src")
sys.path.insert(0, MODEL_SRC_DIR)

from data_loader import load_cifar10
from train import train_model
from evaluate import evaluate_model


def plot_training_history(history):
    """
    Ritar upp tränings- och valideringskurvor (loss och accuracy per epok)
    från ett Keras History-objekt, precis som i notebookens Sektion 7.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Träning")
    axes[0].plot(history.history["val_loss"], label="Validering")
    axes[0].set_title("Loss per epok")
    axes[0].set_xlabel("Epok")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Träning")
    axes[1].plot(history.history["val_accuracy"], label="Validering")
    axes[1].set_title("Accuracy per epok")
    axes[1].set_xlabel("Epok")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    """
    Kör hela pipelinen i ordning: ladda data -> träna -> utvärdera.
    """
    print("Laddar CIFAR-10-datasetet...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10()
    print(
        f"Data laddad: {x_train.shape[0]} träningsbilder, "
        f"{x_val.shape[0]} valideringsbilder, {x_test.shape[0]} testbilder."
    )

    print("\nStartar träning...")
    model, history = train_model(x_train, y_train, x_val, y_val)
    print("Träning klar.")

    print("\nRitar upp tränings- och valideringskurvor...")
    plot_training_history(history)

    print("\nUtvärderar modellen mot testsetet...")
    test_loss, test_accuracy, _ = evaluate_model(model, x_test, y_test)
    print(
        f"\nResultat: test-accuracy {test_accuracy * 100:.2f}%, "
        f"test-loss {test_loss:.4f}"
    )


if __name__ == "__main__":
    main()