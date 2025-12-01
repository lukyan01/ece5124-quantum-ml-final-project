from models.quntum_qnn import QuantumDigitClassifier
from utils.datasets import get_mnist_loaders
from utils.plotting import plot_training_curves, plot_confusion_matrix
from utils.training import train_and_evaluate


def main():
    # currently, quantum circuits run on CPU
    device = "cpu"
    print(f"Using device: {device} (quantum simulation)")

    train_loader, val_loader = get_mnist_loaders(batch_size=32)

    model = QuantumDigitClassifier(
        num_qubits=4,
        num_classes=10,
        input_dim=4,
    )

    model, history, metrics = train_and_evaluate(
        model,
        train_loader,
        val_loader,
        device=device,
        num_epochs=3,
        lr=1e-2,
        run_name="quantum",
    )

    plot_training_curves(history, run_name="quantum", out_dir="run")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=[str(i) for i in range(10)],
        run_name="quantum",
        out_dir="run",
    )


if __name__ == "__main__":
    main()
