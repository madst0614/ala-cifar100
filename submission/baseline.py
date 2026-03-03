"""Part 1: Baseline — ResNet-18 + CIFAR-100 + standard CE loss, 200 epochs.

과제 스펙:
- SGD momentum=0.9, wd=5e-4, lr=0.1, cosine annealing
- batch_size=128
- Train 40k / Val 10k / Test 10k (seed=0)

출력:
- results/curves/baseline_loss.png  — train loss curve
- results/curves/baseline_acc.png   — test accuracy curve
- 최종 test accuracy 출력
"""

import os

import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import set_seed, get_dataloaders, get_model, get_optimizer_and_scheduler

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

TOTAL_EPOCHS = 200
NUM_CLASSES = 100


# ---------------------------------------------------------------------------
# 평가 함수
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """DataLoader 기반 정확도 측정 (%)."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train_baseline():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("results/curves", exist_ok=True)

    # 데이터 로딩
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # 모델, 옵티마이저, 스케줄러
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, T_max=TOTAL_EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()

    # 로그 저장용
    train_losses = []
    val_accs = []
    test_accs = []

    best_val_acc = 0.0
    best_test_acc = 0.0

    print(f"\n{'=' * 60}")
    print(f"Baseline: CE loss, {TOTAL_EPOCHS} epochs")
    print(f"{'=' * 60}")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # --- 학습 ---
        model.train()
        running_loss = 0.0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = ce_loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / total

        # --- 평가 ---
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)

        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{TOTAL_EPOCHS} | "
                f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
            )

    # --- 결과 출력 ---
    print(f"\n{'=' * 60}")
    print(f"Baseline 결과:")
    print(f"  최종 Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"  최고 Test Accuracy: {best_test_acc:.2f}% (best val epoch)")
    print(f"{'=' * 60}")

    # --- Plot: Train Loss ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, TOTAL_EPOCHS + 1), train_losses, "b-", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Baseline: Train Loss (CE)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/curves/baseline_loss.png", dpi=150)
    plt.close()

    # --- Plot: Test Accuracy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, TOTAL_EPOCHS + 1), test_accs, "r-", alpha=0.8, label="Test")
    ax.plot(range(1, TOTAL_EPOCHS + 1), val_accs, "g--", alpha=0.6, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Baseline: Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/curves/baseline_acc.png", dpi=150)
    plt.close()

    print("Plot 저장 완료: results/curves/baseline_loss.png, baseline_acc.png")

    # --- 로그 저장 (train_ala.py에서 비교용으로 사용) ---
    torch.save({
        "train_losses": train_losses,
        "val_accs": val_accs,
        "test_accs": test_accs,
        "final_test_acc": test_accs[-1],
        "best_test_acc": best_test_acc,
    }, "results/baseline_log.pt")
    print("로그 저장: results/baseline_log.pt")


if __name__ == "__main__":
    train_baseline()
