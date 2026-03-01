"""Sanity check: AdaptiveLoss with Φ=I (no RL), otherwise identical to baseline."""

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    set_seed,
    get_dataloaders,
    get_model,
    evaluate,
    get_optimizer_and_scheduler,
)
from adaptive_loss import AdaptiveLoss


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
    return running_loss / total


def main() -> None:
    """Train ResNet-18 on CIFAR-100 with AdaptiveLoss (Φ=I, no RL)."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders()
    model = get_model(device)
    num_classes = 100
    criterion = AdaptiveLoss(num_classes=num_classes).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)

    train_losses: list[float] = []
    test_accs: list[float] = []

    os.makedirs("results", exist_ok=True)
    log_file = open("results/sanity_training_log.txt", "w")

    for epoch in tqdm(range(1, 201), desc="Sanity Check"):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(loss)
        test_accs.append(acc)

        msg = f"Epoch {epoch:3d} | Train Loss: {loss:.4f} | Test Acc: {acc:.2f}%"
        tqdm.write(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # Save results
    os.makedirs("results/curves", exist_ok=True)

    plt.figure()
    plt.plot(range(1, 201), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Sanity Check Train Loss")
    plt.savefig("results/curves/sanity_train_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(range(1, 201), test_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Sanity Check Test Accuracy")
    plt.savefig("results/curves/sanity_test_acc.png", dpi=150)
    plt.close()

    final_msg = f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%"
    print(final_msg)
    log_file.write(final_msg + "\n")
    log_file.close()


if __name__ == "__main__":
    main()
