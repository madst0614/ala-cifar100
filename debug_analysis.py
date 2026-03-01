"""Debug analysis: 3 experiments to diagnose adaptive loss + RL behavior."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import set_seed, get_dataloaders, get_model, get_optimizer_and_scheduler, evaluate
from adaptive_loss import AdaptiveLoss


def experiment1_gradient_norms(device: torch.device) -> None:
    """Compare CE vs AdaptiveLoss gradient norms on a freshly initialized model."""
    print("=" * 60)
    print("Experiment 1: Gradient Norm Comparison (CE vs Adaptive)")
    print("=" * 60)

    set_seed(42)
    model = get_model(device)
    train_loader, _, _ = get_dataloaders()
    ce = nn.CrossEntropyLoss()
    ada = AdaptiveLoss(100).to(device)

    ce_norms = []
    ada_norms = []
    ce_losses = []
    ada_losses = []
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= 10:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        # CE gradient
        model.zero_grad()
        loss_ce = ce(model(inputs), targets)
        loss_ce.backward()
        ce_norm = sum(
            p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5
        ce_norms.append(ce_norm.item())
        ce_losses.append(loss_ce.item())

        # Adaptive gradient (same model state)
        model.zero_grad()
        loss_ada = ada(model(inputs), targets)
        loss_ada.backward()
        ada_norm = sum(
            p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5
        ada_norms.append(ada_norm.item())
        ada_losses.append(loss_ada.item())

    print(f"CE loss:       mean={np.mean(ce_losses):.4f}")
    print(f"Adaptive loss: mean={np.mean(ada_losses):.4f}")
    print(f"CE grad norm:       mean={np.mean(ce_norms):.4f}")
    print(f"Adaptive grad norm: mean={np.mean(ada_norms):.4f}")
    print(f"Ratio (CE/Ada): {np.mean(ce_norms) / np.mean(ada_norms):.1f}x")
    print()


def experiment2_warmup_transition(device: torch.device) -> None:
    """CE warmup 10 epochs then switch to AdaptiveLoss Φ=I (no RL)."""
    print("=" * 60)
    print("Experiment 2: CE Warmup → AdaptiveLoss Transition (no RL)")
    print("=" * 60)

    set_seed(42)
    model = get_model(device)
    train_loader, _, test_loader = get_dataloaders()
    optimizer, scheduler = get_optimizer_and_scheduler(model, T_max=20)

    ce = nn.CrossEntropyLoss()
    ada = AdaptiveLoss(100).to(device)

    for epoch in tqdm(range(1, 21), desc="Exp2"):
        criterion = ce if epoch <= 10 else ada
        phase = "CE" if epoch <= 10 else "ADA"

        model.train()
        running_loss = 0.0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / total
        print(f"  Epoch {epoch:2d} [{phase}] | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")

    print()


def experiment3_inner_distribution(device: torch.device) -> None:
    """Inner product distribution: Φ=I vs Φ perturbed, after CE training."""
    print("=" * 60)
    print("Experiment 3: Inner Product Distribution (Φ=I vs Φ perturbed)")
    print("=" * 60)

    set_seed(42)
    model = get_model(device)
    train_loader, val_loader, _ = get_dataloaders()
    optimizer, scheduler = get_optimizer_and_scheduler(model, T_max=10)

    ce = nn.CrossEntropyLoss()

    # Train 10 epochs with CE
    print("  Training 10 epochs with CE...")
    for epoch in tqdm(range(1, 11), desc="Exp3 CE"):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = ce(model(inputs), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Compute inner product distributions on val set
    model.eval()
    phi_I = torch.eye(100, device=device)

    set_seed(123)
    phi_perturbed = torch.eye(100, device=device)
    phi_perturbed += 0.1 * torch.randn(100, 100, device=device)
    phi_perturbed.fill_diagonal_(1.0)

    all_inner_clean = []
    all_inner_perturbed = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            log_probs = F.log_softmax(logits, dim=1)
            y = F.one_hot(targets, 100).float()

            # Φ=I
            inner_clean = ((y @ phi_I) * log_probs).sum(dim=1)
            all_inner_clean.append(inner_clean)

            # Φ perturbed
            inner_pert = ((y @ phi_perturbed) * log_probs).sum(dim=1)
            all_inner_perturbed.append(inner_pert)

    clean = torch.cat(all_inner_clean)
    pert = torch.cat(all_inner_perturbed)

    print(f"  Φ=I inner:       mean={clean.mean():.2f}, std={clean.std():.2f}, "
          f"min={clean.min():.2f}, max={clean.max():.2f}")
    print(f"  sigmoid(inner):  mean={torch.sigmoid(clean).mean():.4f}, "
          f"std={torch.sigmoid(clean).std():.4f}")
    print(f"  Φ perturbed:     mean={pert.mean():.2f}, std={pert.std():.2f}, "
          f"min={pert.min():.2f}, max={pert.max():.2f}")
    print(f"  sigmoid(inner):  mean={torch.sigmoid(pert).mean():.4f}, "
          f"std={torch.sigmoid(pert).std():.4f}")
    print()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    experiment1_gradient_norms(device)
    experiment2_warmup_transition(device)
    experiment3_inner_distribution(device)

    print("=" * 60)
    print("All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
