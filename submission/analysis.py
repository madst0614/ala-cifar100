"""Part 5: Visualization and analysis.

Reads results from train_ala.py (results/train_ala_results.pt) and generates:
(a) Phi heatmap: 4-subplot at epochs 50, 100, 150, 200
(b) Train loss + Test accuracy curve (baseline vs ALA)
(c) Policy entropy curve
(d) Confusion-Phi correlation curve

All plots -> results/curves/
"""

import os
import glob

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results():
    """Load result data saved by train_ala.py."""
    path = "results/train_ala_results.pt"
    assert os.path.exists(path), (
        f"{path} not found -- run python train_ala.py first"
    )
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# (a) Phi Heatmap: 4-subplot (epoch 50, 100, 150, 200)
# ---------------------------------------------------------------------------

def plot_phi_heatmaps(results):
    """Generate per-experiment Phi heatmap subplots with auto-scaled colorbar."""
    snapshot_epochs = [50, 100, 150, 200]

    # ALA experiment names (exclude baseline)
    ala_names = [name for name in results if name not in ("baseline_ce", "warmup_log")]

    for name in ala_names:
        # --- Pass 1: find global abs_max across all snapshots for this experiment ---
        global_abs_max = 0.0
        phi_data_cache = {}
        for epoch in snapshot_epochs:
            phi_path = f"results/phi_heatmaps/{name}_phi_epoch{epoch}.pt"
            if os.path.exists(phi_path):
                phi = torch.load(phi_path, map_location="cpu", weights_only=True)
                phi_np = phi.numpy()
                mask = np.eye(phi_np.shape[0], dtype=bool)
                off_diag = phi_np[~mask]
                abs_max = np.abs(off_diag).max()
                if abs_max > global_abs_max:
                    global_abs_max = abs_max
                phi_data_cache[epoch] = phi_np

        # Fallback to avoid vmin=vmax=0
        if global_abs_max < 1e-12:
            global_abs_max = 1e-6

        # --- Pass 2: plot with consistent color scale ---
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        fig.suptitle(f"{name}: Phi Evolution (epoch 50-200)", fontsize=14)

        im = None
        for idx, epoch in enumerate(snapshot_epochs):
            ax = axes[idx]

            if epoch in phi_data_cache:
                phi_np = phi_data_cache[epoch]
                mask = np.eye(phi_np.shape[0], dtype=bool)
                phi_display = phi_np.copy()
                phi_display[mask] = 0.0

                im = ax.imshow(phi_display, cmap="RdBu_r",
                               vmin=-global_abs_max, vmax=global_abs_max, aspect="auto")
                ax.set_title(f"Epoch {epoch}")

                # Off-diagonal stats
                off_diag = phi_np[~mask]
                ax.text(0.02, 0.98,
                        f"|mean|={np.abs(off_diag).mean():.4f}\nmax={np.abs(off_diag).max():.4f}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, f"No data\n(epoch {epoch})",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"Epoch {epoch}")

            ax.set_xlabel("Class j")
            if idx == 0:
                ax.set_ylabel("Class i")

        if im is not None:
            plt.colorbar(im, ax=axes, shrink=0.8, label="Phi off-diagonal value")
        plt.tight_layout()
        path = f"results/curves/{name}_phi_evolution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved phi heatmap: {path}")


# ---------------------------------------------------------------------------
# (b) Train Loss + Test Accuracy comparison
# ---------------------------------------------------------------------------

def plot_comparison_curves(results):
    """Loss/accuracy comparison: Baseline CE vs ALA experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    warmup_log = results.get("warmup_log", [])
    warmup_epochs = [e["epoch"] for e in warmup_log]
    warmup_losses = [e["train_loss"] for e in warmup_log]
    warmup_test = [e["test_acc"] for e in warmup_log]

    colors = {"baseline_ce": "gray", "spec_faithful": "tab:red", "stabilized": "tab:blue"}

    for name, data in results.items():
        if name == "warmup_log":
            continue
        elogs = data.get("epoch_logs", [])
        if not elogs:
            continue

        color = colors.get(name, None)
        epochs = [e["epoch"] for e in elogs]
        losses = [e["train_loss"] for e in elogs]
        test_accs = [e["test_acc"] for e in elogs]

        # Loss curve (warmup + experiment)
        ax = axes[0]
        if warmup_log:
            ax.plot(warmup_epochs, warmup_losses, color=color, alpha=0.3, linestyle="--")
        ax.plot(epochs, losses, label=name, color=color, alpha=0.8)

        # Test accuracy curve
        ax = axes[1]
        if warmup_log:
            ax.plot(warmup_epochs, warmup_test, color=color, alpha=0.3, linestyle="--")
        ax.plot(epochs, test_accs, label=name, color=color, alpha=0.8)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Train Loss: Baseline CE vs ALA")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("Test Accuracy: Baseline CE vs ALA")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/curves/comparison_loss_acc.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved comparison plot: {path}")


# ---------------------------------------------------------------------------
# (c) Policy Entropy Curve
# ---------------------------------------------------------------------------

def plot_policy_entropy(results):
    """Policy entropy over training steps for ALA experiments."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"spec_faithful": "tab:red", "stabilized": "tab:blue"}

    for name, data in results.items():
        if name in ("baseline_ce", "warmup_log"):
            continue
        wlogs = data.get("window_logs", [])
        if not wlogs:
            continue

        steps = [w["step"] for w in wlogs]
        entropy = [w["entropy"] for w in wlogs]
        color = colors.get(name, None)
        ax.plot(steps, entropy, label=name, color=color, alpha=0.8)

    max_entropy = np.log(3)
    ax.axhline(y=max_entropy, color="gray", linestyle=":", alpha=0.5,
               label=f"max entropy (ln3={max_entropy:.3f})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy (Exploration Monitoring)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/curves/policy_entropy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved entropy plot: {path}")


# ---------------------------------------------------------------------------
# (d) Confusion-Phi Correlation Curve
# ---------------------------------------------------------------------------

def plot_confusion_phi_correlation(results):
    """Off-diagonal Pearson correlation between confusion matrix and Phi."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"spec_faithful": "tab:red", "stabilized": "tab:blue"}

    for name, data in results.items():
        if name in ("baseline_ce", "warmup_log"):
            continue
        wlogs = data.get("window_logs", [])
        if not wlogs:
            continue

        steps = [w["step"] for w in wlogs]
        corr = [w["corr"] for w in wlogs]
        color = colors.get(name, None)
        ax.plot(steps, corr, label=name, color=color, alpha=0.8)

    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Confusion-Phi Correlation (off-diagonal)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/curves/confusion_phi_corr.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved correlation plot: {path}")


# ---------------------------------------------------------------------------
# Extra: Phi abs_mean movement plot
# ---------------------------------------------------------------------------

def plot_phi_movement(results):
    """Phi off-diagonal |mean| over time -- how much the RL controller adjusted Phi."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"spec_faithful": "tab:red", "stabilized": "tab:blue"}

    for name, data in results.items():
        if name in ("baseline_ce", "warmup_log"):
            continue
        wlogs = data.get("window_logs", [])
        if not wlogs:
            continue

        steps = [w["step"] for w in wlogs]
        phi_am = [w["phi_abs_mean"] for w in wlogs]
        color = colors.get(name, None)
        ax.plot(steps, phi_am, label=name, color=color, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Phi Off-Diag |mean|")
    ax.set_title("Phi Movement (|mean| off-diagonal)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/curves/phi_movement.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved phi movement plot: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("results/curves", exist_ok=True)
    os.makedirs("results/phi_heatmaps", exist_ok=True)

    print("Loading result data...")
    results = load_results()

    print("\nGenerating visualizations...")

    # (a) Phi heatmap evolution
    print("\n(a) Phi heatmap:")
    plot_phi_heatmaps(results)

    # (b) Loss/Accuracy comparison
    print("\n(b) Loss/Accuracy comparison:")
    plot_comparison_curves(results)

    # (c) Policy entropy
    print("\n(c) Policy entropy:")
    plot_policy_entropy(results)

    # (d) Confusion-Phi correlation
    print("\n(d) Confusion-Phi correlation:")
    plot_confusion_phi_correlation(results)

    # Extra: Phi movement
    print("\n(+) Phi movement:")
    plot_phi_movement(results)

    print("\nAll visualizations complete.")
    print("Check: results/curves/, results/phi_heatmaps/")


if __name__ == "__main__":
    main()
