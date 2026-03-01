"""Part 5a: Phi heatmap visualization across training checkpoints."""

import os

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Load saved Phi matrices and visualize their evolution as heatmaps."""
    os.makedirs("results/phi_heatmaps", exist_ok=True)

    checkpoints = [50, 100, 150, 200]
    phis: list[torch.Tensor] = []

    for epoch in checkpoints:
        path = f"results/phi_heatmaps/phi_epoch_{epoch:03d}.pt"
        phi = torch.load(path, map_location="cpu", weights_only=True)
        phis.append(phi)

    # 2x2 subplot heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (epoch, phi) in enumerate(zip(checkpoints, phis)):
        ax = axes[idx]
        phi_np = phi.numpy()
        im = ax.imshow(phi_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"Epoch {epoch}")
        ax.set_xlabel("Class j")
        ax.set_ylabel("Class i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Phi Matrix Evolution", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/phi_heatmaps/phi_evolution.png", dpi=150)
    plt.close()

    # Print statistics
    print("=" * 60)
    print("Phi Matrix Statistics")
    print("=" * 60)

    for epoch, phi in zip(checkpoints, phis):
        phi_np = phi.numpy()
        num_classes = phi_np.shape[0]
        diag = np.diag(phi_np)
        mask = ~np.eye(num_classes, dtype=bool)
        off_diag = phi_np[mask]

        print(f"\nEpoch {epoch}:")
        print(f"  Diagonal   — mean: {diag.mean():.4f}, std: {diag.std():.4f}, "
              f"min: {diag.min():.4f}, max: {diag.max():.4f}")
        print(f"  Off-diag   — mean: {off_diag.mean():.4f}, std: {off_diag.std():.4f}, "
              f"min: {off_diag.min():.4f}, max: {off_diag.max():.4f}")
        print(f"  Symmetry check: max|Phi - Phi^T| = {np.abs(phi_np - phi_np.T).max():.2e}")


if __name__ == "__main__":
    main()
