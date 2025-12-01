import torch

from models.classical_cnn import ClassicalMNISTCNN
from utils.datasets import get_mnist_loaders
from utils.plotting import plot_training_curves, plot_confusion_matrix
from utils.training import train_and_evaluate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader = get_mnist_loaders(batch_size=64)

    model = ClassicalMNISTCNN(num_classes=10)

    model, history, metrics = train_and_evaluate(
        model,
        train_loader,
        val_loader,
        device=device,
        num_epochs=5,
        lr=1e-3,
        run_name="classical",
    )

    # Plots
    plot_training_curves(history, run_name="classical", out_dir="run")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=[str(i) for i in range(10)],
        run_name="classical",
        out_dir="run",
    )


if __name__ == "__main__":
    main()
