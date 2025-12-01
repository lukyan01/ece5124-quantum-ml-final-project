import copy
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from utils.metrics import compute_classification_metrics


def evaluate_model(
        model: nn.Module,
        data_loader,
        device: str = "cpu",
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on data_loader. Returns (loss, accuracy, y_true, y_pred).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            all_true.extend(targets.cpu().numpy().tolist())
            all_pred.extend(predicted.cpu().numpy().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_true), np.array(all_pred)


def train_and_evaluate(
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = "cuda",
        num_epochs: int = 5,
        lr: float = 1e-3,
        run_name: str = "model",
) -> Tuple[nn.Module, Dict[str, List[float]], Dict]:
    """
    Trains the model and evaluates on validation set.

    Returns: (best_model, history, metrics_dict)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"{run_name} Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix(
                loss=running_loss / total,
                acc=correct / total,
            )

        train_loss = running_loss / total
        val_loss, val_acc, y_true, y_pred = evaluate_model(model, val_loader, device=device)
        epoch_time = time.perf_counter() - epoch_start

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"[{run_name}] Epoch {epoch + 1}/{num_epochs} "
            f"- Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
            f"Val acc: {val_acc:.4f}, Time: {epoch_time:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # load best weights
    model.load_state_dict(best_state)

    # final eval
    val_loss, val_acc, y_true, y_pred = evaluate_model(model, val_loader, device=device)
    metrics = compute_classification_metrics(y_true, y_pred, average="macro")
    metrics["val_loss"] = val_loss
    metrics["val_accuracy"] = val_acc

    print(f"[{run_name}] Final metrics:")
    for k in ["accuracy", "precision", "recall", "f1", "val_loss", "val_accuracy"]:
        print(f"  {k}: {metrics[k]:.4f}")

    return model, history, metrics
