import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_one_epoch(model, loader, optimizer, criterion, device):
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


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        ),
    ])

    # Datasets
    full_train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform,
    )
    train_dataset, val_dataset = random_split(
        full_train_dataset, [40000, 10000],
        generator=torch.Generator().manual_seed(0),
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2,
    )

    # Model
    model = models.resnet18(num_classes=100).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training loop
    train_losses = []
    test_accs = []

    for epoch in tqdm(range(1, 201), desc="Training"):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(loss)
        test_accs.append(acc)

        tqdm.write(
            f"Epoch {epoch:3d} | Train Loss: {loss:.4f} | Test Acc: {acc:.2f}%"
        )

    # Save results
    os.makedirs("results/curves", exist_ok=True)

    plt.figure()
    plt.plot(range(1, 201), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Baseline Train Loss")
    plt.savefig("results/curves/baseline_train_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(range(1, 201), test_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Baseline Test Accuracy")
    plt.savefig("results/curves/baseline_test_acc.png", dpi=150)
    plt.close()

    print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%")


if __name__ == "__main__":
    main()
