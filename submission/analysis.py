"""Part 5: 시각화 및 분석.

train_ala.py의 결과 데이터(results/train_ala_results.pt)를 읽어서:
(a) Phi heatmap: epoch 50,100,150,200 시점 4개 subplot
(b) Train loss + Test accuracy curve (baseline vs ALA 비교)
(c) Policy entropy curve
(d) Confusion-Phi correlation curve

모든 plot → results/curves/
"""

import os
import glob

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 결과 로딩
# ---------------------------------------------------------------------------

def load_results():
    """train_ala.py에서 저장한 결과 데이터 로드."""
    path = "results/train_ala_results.pt"
    assert os.path.exists(path), (
        f"{path} 없음 — 먼저 python train_ala.py 실행 필요"
    )
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# (a) Phi Heatmap: 4-subplot (epoch 50, 100, 150, 200)
# ---------------------------------------------------------------------------

def plot_phi_heatmaps(results):
    """ALA 실험들의 Phi heatmap을 epoch별 subplot으로 생성."""
    snapshot_epochs = [50, 100, 150, 200]

    # ALA 실험 이름 목록 (baseline 제외)
    ala_names = [name for name in results if name not in ("baseline_ce", "warmup_log")]

    for name in ala_names:
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        fig.suptitle(f"{name}: Phi 변화 (epoch 50 → 200)", fontsize=14)

        for idx, epoch in enumerate(snapshot_epochs):
            ax = axes[idx]
            phi_path = f"results/phi_heatmaps/{name}_phi_epoch{epoch}.pt"

            if os.path.exists(phi_path):
                phi = torch.load(phi_path, map_location="cpu", weights_only=True)
                phi_np = phi.numpy()
                # 대각선 마스킹 (off-diagonal만 시각화)
                mask = np.eye(phi_np.shape[0], dtype=bool)
                phi_display = phi_np.copy()
                phi_display[mask] = 0.0

                im = ax.imshow(phi_display, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
                ax.set_title(f"Epoch {epoch}")

                # off-diagonal 통계
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

        plt.colorbar(im, ax=axes, shrink=0.8, label="Phi off-diagonal value")
        plt.tight_layout()
        path = f"results/curves/{name}_phi_evolution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Phi heatmap 저장: {path}")


# ---------------------------------------------------------------------------
# (b) Train Loss + Test Accuracy 비교
# ---------------------------------------------------------------------------

def plot_comparison_curves(results):
    """Baseline CE vs ALA 실험들의 loss/accuracy 비교 플롯."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Warmup log가 있으면 앞에 추가
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

        # Loss curve (warmup + 실험)
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
    print(f"  비교 플롯 저장: {path}")


# ---------------------------------------------------------------------------
# (c) Policy Entropy Curve
# ---------------------------------------------------------------------------

def plot_policy_entropy(results):
    """ALA 실험들의 정책 엔트로피 변화."""
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

    # 이론적 최대 엔트로피 (3개 액션 균등 분포)
    max_entropy = np.log(3)
    ax.axhline(y=max_entropy, color="gray", linestyle=":", alpha=0.5, label=f"max entropy (ln3={max_entropy:.3f})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy (탐색 정도 모니터링)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/curves/policy_entropy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  엔트로피 플롯 저장: {path}")


# ---------------------------------------------------------------------------
# (d) Confusion-Phi Correlation Curve
# ---------------------------------------------------------------------------

def plot_confusion_phi_correlation(results):
    """혼동행렬과 Phi의 off-diagonal Pearson 상관관계 변화."""
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
    print(f"  상관관계 플롯 저장: {path}")


# ---------------------------------------------------------------------------
# 추가: Phi abs_mean 변화 플롯
# ---------------------------------------------------------------------------

def plot_phi_movement(results):
    """Phi off-diagonal |mean| 변화 — RL controller가 Phi를 얼마나 조정했는지."""
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
    ax.set_title("Phi 변화량 (|mean| of off-diagonal)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/curves/phi_movement.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Phi 변화량 플롯 저장: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("results/curves", exist_ok=True)
    os.makedirs("results/phi_heatmaps", exist_ok=True)

    print("결과 데이터 로딩...")
    results = load_results()

    print("\n시각화 생성 중...")

    # (a) Phi heatmap evolution
    print("\n(a) Phi heatmap:")
    plot_phi_heatmaps(results)

    # (b) Loss/Accuracy 비교
    print("\n(b) Loss/Accuracy 비교:")
    plot_comparison_curves(results)

    # (c) Policy entropy
    print("\n(c) Policy entropy:")
    plot_policy_entropy(results)

    # (d) Confusion-Phi correlation
    print("\n(d) Confusion-Phi correlation:")
    plot_confusion_phi_correlation(results)

    # 추가: Phi movement
    print("\n(+) Phi movement:")
    plot_phi_movement(results)

    print("\n모든 시각화 완료.")
    print("결과 확인: results/curves/, results/phi_heatmaps/")


if __name__ == "__main__":
    main()
