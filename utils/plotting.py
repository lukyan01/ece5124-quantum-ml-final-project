import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
        history: Dict[str, List[float]],
        run_name: str = "model",
        out_dir: str = "./",
):
    """
    Plots training loss + validation loss and accuracy over epochs
    """
    epochs = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curves - {run_name}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{run_name}_loss.png")
    plt.close()

    # Accuracy
    if "val_accuracy" in history:
        plt.figure()
        plt.plot(epochs, history["val_accuracy"], label="Val accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Validation accuracy - {run_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{run_name}_accuracy.png")
        plt.close()


def plot_metrics_bar_compare(
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
        labels: tuple = ("Classical", "Quantum"),
        out_path: str = "./model_compare_metrics.png",
):
    """
    Simple bar chart comparing accuracy, precision, recall, F1
    """
    keys = ["accuracy", "precision", "recall", "f1"]
    a_vals = [metrics_a[k] for k in keys]
    b_vals = [metrics_b[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, a_vals, width, label=labels[0])
    plt.bar(x + width / 2, b_vals, width, label=labels[1])

    plt.xticks(x, [k.capitalize() for k in keys])
    plt.ylabel("Score")
    plt.title("Model metrics comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, run_name: str, out_dir: str = "./"):
    """
    Basic confusion matrix heatmap
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {run_name}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{run_name}_confusion_matrix.png")
    plt.close()
