"""Common utilities for ALA CIFAR-100 experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(
    batch_size: int = 128, num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-100 and return train/val/test dataloaders.

    Train set (50k) is split into 40k train / 10k val.
    Train has augmentation; val and test have Normalize only.
    Two separate datasets are created with the same seed split so that
    train indices get augmentation and val indices do not.
    """
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Augmented full dataset → take train split
    train_full = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform,
    )
    train_dataset, _ = random_split(
        train_full, [40000, 10000],
        generator=torch.Generator().manual_seed(0),
    )

    # Non-augmented full dataset → take val split
    val_full = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=test_transform,
    )
    _, val_dataset = random_split(
        val_full, [40000, 10000],
        generator=torch.Generator().manual_seed(0),
    )

    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_model(device: torch.device, num_classes: int = 100) -> nn.Module:
    """Create a ResNet-18 model for CIFAR-100."""
    model = models.resnet18(num_classes=num_classes)
    return model.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate model accuracy on a dataloader.

    Returns accuracy as a float in range 0-100.
    """
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


def get_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    T_max: int = 200,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """Create SGD optimizer and CosineAnnealingLR scheduler."""
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return optimizer, scheduler
