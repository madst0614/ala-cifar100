"""Part 4(c): ALA 전체 학습 파이프라인.

3단계로 실행:
  Phase 0: CE warmup 50 epochs → checkpoint 저장
  Phase 1: baseline_ce — CE only 150 more epochs (200 total)
  Phase 2: ALA paper_spec — beta=0.1, sqrt normalization, clamp[-1,1], sign reward
  Phase 3: ALA stabilized — beta=0.1 * delta_scale=0.01, sqrt normalization, clamp[-1,1], sign reward

핵심 파라미터 (과제 스펙):
  K=200 (window size), gamma=0.9, policy_lr=0.001
  replay memory=1000, policy_batch_size=8
  single child model (논문과 동일)

출력:
  results/summary.txt              — 최종 비교 테이블
  results/phi_heatmaps/*.png       — Phi heatmap (epoch 50,100,150,200)
  results/curves/comparison_*.png  — 비교 플롯
  results/train_ala_results.pt     — analysis.py용 결과 데이터
"""

import math
import os
import traceback
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    set_seed, get_dataloaders, get_model, get_optimizer_and_scheduler,
    fast_evaluate, fast_confusion_matrix,
)
from adaptive_loss import AdaptiveLoss
from state import construct_states, get_pair_indices_tensor
from controller import ALAPolicy, ReplayMemory, compute_reward

# ---------------------------------------------------------------------------
# 실험 설정
# ---------------------------------------------------------------------------

CONFIGS = OrderedDict({
    "baseline_ce": {
        "description": "CE only (no ALA), warmup 이후 150 epochs 추가 학습",
        "use_ala": False,
    },
    "paper_spec": {
        "description": "ALA 논문 스펙 그대로: beta=0.1, sqrt norm, clamp[-1,1], sign reward",
        "use_ala": True,
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 1.0,  # beta * delta_scale = 0.1 * 1.0 = 0.1 (논문 스펙)
        "reward_type": "sign",
    },
    "stabilized": {
        "description": "안정화 버전: beta=0.1 * delta_scale=0.01 = step 0.001, sqrt norm",
        "use_ala": True,
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,  # beta * delta_scale = 0.1 * 0.01 = 0.001
        "reward_type": "sign",
    },
})

NUM_CLASSES = 100
WARMUP_EPOCHS = 50
TOTAL_EPOCHS = 200
POST_WARMUP_EPOCHS = 150  # = TOTAL_EPOCHS - WARMUP_EPOCHS
BETA = 0.1
K = 200        # window size
GAMMA = 0.9    # discount factor
POLICY_BATCH_SIZE = 8

# Phi heatmap 저장 epoch (warmup 포함 실제 epoch 기준)
PHI_SNAPSHOT_EPOCHS = {50, 100, 150, 200}


# ---------------------------------------------------------------------------
# 헬퍼: actions → delta_phi
# ---------------------------------------------------------------------------

def actions_to_delta_phi(actions, pair_i, pair_j, num_classes, beta, device):
    """액션 인덱스를 Phi 업데이트 행렬로 변환.

    action 0 = -beta, 1 = 0, 2 = +beta.
    """
    action_map = torch.tensor([-beta, 0.0, beta], device=device)
    delta_values = action_map[actions]
    delta_phi = torch.zeros(num_classes, num_classes, device=device)
    delta_phi[pair_i, pair_j] = delta_values
    delta_phi[pair_j, pair_i] = delta_values
    return delta_phi


# ---------------------------------------------------------------------------
# 헬퍼: REINFORCE 정책 업데이트
# ---------------------------------------------------------------------------

def update_policy(policy, optimizer, memory, batch_size, baseline_ema, device):
    """REINFORCE with baseline으로 정책 네트워크 업데이트.

    외부 RL 라이브러리 없이 직접 구현.
    """
    samples = memory.sample(batch_size)
    total_loss = torch.tensor(0.0, device=device)
    total_reward = 0.0
    for states, actions, _old_lp, reward in samples:
        states = states.to(device)
        actions = actions.to(device)
        logits = policy(states)
        dist = Categorical(logits=logits)
        new_lp = dist.log_prob(actions)
        advantage = reward - baseline_ema
        total_loss = total_loss + (-(new_lp * advantage).mean())
        total_reward += reward
    total_loss = total_loss / len(samples)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    avg_reward = total_reward / len(samples)
    baseline_ema = 0.9 * baseline_ema + 0.1 * avg_reward
    return baseline_ema


# ---------------------------------------------------------------------------
# 헬퍼: 정책 엔트로피
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_policy_entropy(policy, states):
    """정책의 액션 분포 엔트로피 계산 (탐색 정도 모니터링)."""
    logits = policy(states)
    dist = Categorical(logits=logits)
    return dist.entropy().mean().item()


# ---------------------------------------------------------------------------
# 헬퍼: Confusion-Phi Pearson 상관관계 (off-diagonal)
# ---------------------------------------------------------------------------

def confusion_phi_correlation(confusion, phi, num_classes):
    """혼동행렬과 Phi의 off-diagonal 요소 간 Pearson 상관계수."""
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=phi.device)
    c_off = confusion[mask].float()
    p_off = phi[mask].float()
    if c_off.std() < 1e-12 or p_off.std() < 1e-12:
        return 0.0
    c_centered = c_off - c_off.mean()
    p_centered = p_off - p_off.mean()
    corr = (c_centered * p_centered).sum() / (c_centered.norm() * p_centered.norm() + 1e-12)
    return corr.item()


# ---------------------------------------------------------------------------
# 헬퍼: Phi heatmap 저장
# ---------------------------------------------------------------------------

def save_phi_heatmap(phi_data, name, epoch):
    """Save Phi off-diagonal heatmap with data-driven color scale."""
    fig, ax = plt.subplots(figsize=(8, 7))
    phi_np = phi_data.cpu().numpy()
    mask = np.eye(phi_np.shape[0], dtype=bool)
    phi_display = phi_np.copy()
    phi_display[mask] = 0.0

    # Auto-scale: use actual data range instead of fixed [-1, 1]
    abs_max = max(np.abs(phi_np[~mask]).max(), 1e-6)
    im = ax.imshow(phi_display, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    ax.set_title(f"{name}: Phi at epoch {epoch}")
    ax.set_xlabel("Class j")
    ax.set_ylabel("Class i")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = f"results/phi_heatmaps/{name}_phi_epoch{epoch}.png"
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Phase 0: CE Warmup
# ---------------------------------------------------------------------------

def run_warmup(device, train_loader, val_images, val_targets, test_images, test_targets):
    """CE loss로 50 epoch warmup 후 checkpoint 저장."""
    print("=" * 70)
    print("Phase 0: CE Warmup (50 epochs)")
    print("=" * 70)

    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, T_max=TOTAL_EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()

    warmup_log = []

    for epoch in range(1, WARMUP_EPOCHS + 1):
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
        val_acc = fast_evaluate(model, val_images, val_targets)
        test_acc = fast_evaluate(model, test_images, test_targets)

        warmup_log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "test_acc": test_acc,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{WARMUP_EPOCHS} | "
                f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
            )

    # Checkpoint 저장
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": WARMUP_EPOCHS,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }
    ckpt_path = "results/warmup_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"\nWarmup 완료. Val: {val_acc:.2f}%, Test: {test_acc:.2f}%")
    print(f"Checkpoint 저장: {ckpt_path}")
    return ckpt_path, warmup_log


# ---------------------------------------------------------------------------
# Phase 1-3: 개별 실험 실행
# ---------------------------------------------------------------------------

def run_experiment(
    name, config, ckpt_path, device,
    train_loader, val_images, val_targets, test_images, test_targets,
    warmup_val_acc=0.0,
):
    """Warmup checkpoint에서 출발하여 POST_WARMUP_EPOCHS만큼 학습."""

    print(f"\n{'=' * 70}")
    print(f"실험: {name}")
    print(f"설명: {config['description']}")
    print(f"{'=' * 70}")

    use_ala = config["use_ala"]

    # Checkpoint에서 모델/옵티마이저/스케줄러 복원
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = get_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer, scheduler = get_optimizer_and_scheduler(model, T_max=TOTAL_EPOCHS)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    ce_loss_fn = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)

    epoch_logs = []   # {epoch, train_loss, val_acc, test_acc}
    window_logs = []  # ALA only: {step, entropy, plus_ratio, ...}
    phi_snapshots = {}  # epoch → phi tensor

    pair_i, pair_j = get_pair_indices_tensor(NUM_CLASSES, device)

    if not use_ala:
        # === CE Baseline: 단순 epoch 루프 ===
        for ep in range(1, POST_WARMUP_EPOCHS + 1):
            real_epoch = WARMUP_EPOCHS + ep
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
            val_acc = fast_evaluate(model, val_images, val_targets)
            test_acc = fast_evaluate(model, test_images, test_targets)
            epoch_logs.append({
                "epoch": real_epoch,
                "train_loss": avg_loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
            })
            if ep % 10 == 0 or ep == 1:
                print(
                    f"  Epoch {real_epoch:3d}/{TOTAL_EPOCHS} | "
                    f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
                )

        return {
            "epoch_logs": epoch_logs,
            "window_logs": window_logs,
            "phi_snapshots": phi_snapshots,
            "status": "OK",
        }

    # === ALA 실험 ===
    clamp_range = config["clamp_range"]
    delta_scale = config["delta_scale"]
    reward_type = config["reward_type"]

    adaptive_loss = AdaptiveLoss(NUM_CLASSES, clamp_range=clamp_range).to(device)
    policy = ALAPolicy(state_dim=24).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    memory = ReplayMemory(1000)

    confusion_history = []
    M_old = None
    prev_states = None
    prev_actions = None
    prev_log_probs = None
    baseline_ema = 0.0

    # K-window 학습 루프
    global_step = 0
    total_steps = POST_WARMUP_EPOCHS * steps_per_epoch
    current_epoch = WARMUP_EPOCHS
    epoch_loss = 0.0
    epoch_total = 0
    train_iter = iter(train_loader)
    num_pairs = NUM_CLASSES * (NUM_CLASSES - 1) // 2

    # Warmup 직후 Phi 스냅샷 (epoch 50, Phi=I)
    if WARMUP_EPOCHS in PHI_SNAPSHOT_EPOCHS:
        phi_snapshots[WARMUP_EPOCHS] = adaptive_loss.phi.data.clone().cpu()
        save_phi_heatmap(adaptive_loss.phi.data, name, WARMUP_EPOCHS)

    pbar = tqdm(total=total_steps, desc=f"  {name}", unit="step", leave=False)

    while global_step < total_steps:
        # === K-step window 시작 ===
        window_start_error = 100.0 - fast_evaluate(model, val_images, val_targets)

        for j in range(1, K + 1):
            try:
                inputs, targets_batch = next(train_iter)
            except StopIteration:
                # Epoch 경계
                current_epoch += 1
                avg_loss = epoch_loss / max(epoch_total, 1)
                val_acc = fast_evaluate(model, val_images, val_targets)
                test_acc = fast_evaluate(model, test_images, test_targets)

                epoch_logs.append({
                    "epoch": current_epoch,
                    "train_loss": avg_loss,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                })

                # Early stopping: warmup 대비 15% 이상 하락 시 중단
                if val_acc < warmup_val_acc - 15.0:
                    print(
                        f"  EARLY STOP at epoch {current_epoch}: "
                        f"val_acc {val_acc:.2f}% < warmup ({warmup_val_acc:.2f}%) - 15%"
                    )
                    pbar.close()
                    return {
                        "epoch_logs": epoch_logs,
                        "window_logs": window_logs,
                        "phi_snapshots": phi_snapshots,
                        "status": f"EARLY_STOP (epoch {current_epoch}, val={val_acc:.1f}%)",
                    }

                ep_rel = current_epoch - WARMUP_EPOCHS
                if ep_rel % 10 == 0 or ep_rel == 1:
                    print(
                        f"  Epoch {current_epoch:3d}/{TOTAL_EPOCHS} | "
                        f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
                    )

                # Phi heatmap 스냅샷
                if current_epoch in PHI_SNAPSHOT_EPOCHS:
                    phi_snapshots[current_epoch] = adaptive_loss.phi.data.clone().cpu()
                    save_phi_heatmap(adaptive_loss.phi.data, name, current_epoch)

                scheduler.step()
                epoch_loss = 0.0
                epoch_total = 0
                train_iter = iter(train_loader)
                inputs, targets_batch = next(train_iter)

            inputs, targets_batch = inputs.to(device), targets_batch.to(device)

            # 모델 학습 스텝 (adaptive loss)
            model.train()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = adaptive_loss(logits, targets_batch)

            # NaN 체크
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at step {global_step}, 실험 중단.")
                pbar.close()
                return {
                    "epoch_logs": epoch_logs,
                    "window_logs": window_logs,
                    "phi_snapshots": phi_snapshots,
                    "status": "FAILED (NaN)",
                }

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_total += inputs.size(0)
            global_step += 1
            pbar.update(1)

            if global_step >= total_steps:
                break

        # === K-step window 종료 ===
        window_end_error = 100.0 - fast_evaluate(model, val_images, val_targets)
        M_new = window_end_error

        # 보상 계산 및 메모리 저장
        reward_val = None
        if M_old is not None and prev_states is not None:
            reward_val = compute_reward(M_old, M_new)
            memory.push(prev_states, prev_actions, prev_log_probs, reward_val)

        # 정책 업데이트
        if len(memory) >= POLICY_BATCH_SIZE:
            baseline_ema = update_policy(
                policy, policy_optimizer, memory,
                POLICY_BATCH_SIZE, baseline_ema, device,
            )

        # 혼동행렬 → 상태 구성 → 액션 → Phi 업데이트
        progress = global_step / total_steps
        C = fast_confusion_matrix(model, val_images, val_targets, NUM_CLASSES)
        confusion_history.append(C)
        if len(confusion_history) > 10:
            confusion_history.pop(0)

        all_states = construct_states(
            confusion_history, adaptive_loss.phi.data, progress,
            NUM_CLASSES, pair_i, pair_j,
        )

        with torch.no_grad():
            actions, log_probs = policy.select_action(all_states)

        delta_phi = actions_to_delta_phi(
            actions, pair_i, pair_j, NUM_CLASSES, BETA, device,
        )
        adaptive_loss.update_phi(delta_phi * delta_scale)

        # === 진단 로그 ===
        entropy = compute_policy_entropy(policy, all_states)

        n_plus = (actions == 2).sum().item()
        n_zero = (actions == 1).sum().item()
        n_minus = (actions == 0).sum().item()
        total_a = actions.numel()

        diag_mask = torch.eye(NUM_CLASSES, dtype=torch.bool, device=device)
        phi_off = adaptive_loss.phi.data[~diag_mask]
        phi_abs_mean = phi_off.abs().mean().item()
        phi_abs_max = phi_off.abs().max().item()

        corr = confusion_phi_correlation(C, adaptive_loss.phi.data, NUM_CLASSES)

        window_logs.append({
            "step": global_step,
            "entropy": entropy,
            "plus_ratio": n_plus / total_a,
            "zero_ratio": n_zero / total_a,
            "minus_ratio": n_minus / total_a,
            "phi_abs_mean": phi_abs_mean,
            "phi_abs_max": phi_abs_max,
            "corr": corr,
            "reward": reward_val if reward_val is not None else 0.0,
        })

        M_old = M_new
        prev_states = all_states
        prev_actions = actions
        prev_log_probs = log_probs

    pbar.close()

    # 마지막 epoch 처리 (미완료분)
    if epoch_total > 0:
        current_epoch += 1
        avg_loss = epoch_loss / epoch_total
        val_acc = fast_evaluate(model, val_images, val_targets)
        test_acc = fast_evaluate(model, test_images, test_targets)
        epoch_logs.append({
            "epoch": current_epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "test_acc": test_acc,
        })
        if current_epoch in PHI_SNAPSHOT_EPOCHS:
            phi_snapshots[current_epoch] = adaptive_loss.phi.data.clone().cpu()
            save_phi_heatmap(adaptive_loss.phi.data, name, current_epoch)

    return {
        "epoch_logs": epoch_logs,
        "window_logs": window_logs,
        "phi_snapshots": phi_snapshots,
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------------------------

def generate_summary(results, warmup_log):
    """비교 테이블 생성 및 results/summary.txt 저장."""
    lines = []
    lines.append("=" * 85)
    lines.append("ALA 실험 결과 비교")
    lines.append("=" * 85)

    header = (
        f"{'실험':<18} | {'최종 Test':>10} | {'최고 Test':>10} | "
        f"{'Entropy':>8} | {'Phi |mean|':>10} | {'Corr':>7} | {'상태':<20}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Warmup 결과
    if warmup_log:
        wl = warmup_log[-1]
        lines.append(
            f"{'warmup (ep50)':<18} | {wl['test_acc']:9.2f}% | {max(e['test_acc'] for e in warmup_log):9.2f}% | "
            f"{'-':>8} | {'-':>10} | {'-':>7} | {'OK':<20}"
        )

    for name, result in results.items():
        elogs = result["epoch_logs"]
        if not elogs:
            lines.append(f"{name:<18} | {'N/A':>10} | {'N/A':>10} | "
                         f"{'-':>8} | {'-':>10} | {'-':>7} | {result['status']:<20}")
            continue

        final_test = elogs[-1]["test_acc"]
        best_test = max(e["test_acc"] for e in elogs)

        wlogs = result["window_logs"]
        if wlogs:
            ent = f"{wlogs[-1]['entropy']:.3f}"
            phi_m = f"{wlogs[-1]['phi_abs_mean']:.6f}"
            corr_s = f"{wlogs[-1]['corr']:+.4f}"
        else:
            ent = "-"
            phi_m = "-"
            corr_s = "-"

        lines.append(
            f"{name:<18} | {final_test:9.2f}% | {best_test:9.2f}% | "
            f"{ent:>8} | {phi_m:>10} | {corr_s:>7} | {result['status']:<20}"
        )

    text = "\n".join(lines)
    print("\n" + text)
    with open("results/summary.txt", "w") as f:
        f.write(text + "\n")
    print("\n결과 저장: results/summary.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("results/phi_heatmaps", exist_ok=True)
    os.makedirs("results/curves", exist_ok=True)

    # 데이터 로딩
    train_loader, val_loader, test_loader = get_dataloaders()

    # GPU에 val/test 데이터 미리 적재 (빠른 평가용)
    print("Val/Test 데이터 GPU 적재 중...")
    val_images = torch.cat([x for x, _ in val_loader]).to(device)
    val_targets = torch.cat([y for _, y in val_loader]).to(device)
    test_images = torch.cat([x for x, _ in test_loader]).to(device)
    test_targets = torch.cat([y for _, y in test_loader]).to(device)
    print(f"  val: {val_images.shape}, test: {test_images.shape}")

    # Phase 0: Warmup
    ckpt_path = "results/warmup_checkpoint.pt"
    if os.path.exists(ckpt_path):
        print(f"\nWarmup checkpoint 발견: {ckpt_path} — warmup 스킵")
        warmup_log = []
    else:
        ckpt_path, warmup_log = run_warmup(
            device, train_loader, val_images, val_targets, test_images, test_targets,
        )

    # Early stopping 기준: warmup val_acc
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    warmup_val_acc = ckpt["val_acc"]
    del ckpt
    print(f"Warmup val_acc (early stopping 기준): {warmup_val_acc:.2f}%")

    # Phase 1-3: 실험 실행
    results = OrderedDict()
    total_exps = len(CONFIGS)
    for idx, (name, config) in enumerate(CONFIGS.items(), 1):
        print(f"\n[{idx}/{total_exps}] 실험 시작: {name}")
        try:
            result = run_experiment(
                name, config, ckpt_path, device,
                train_loader, val_images, val_targets, test_images, test_targets,
                warmup_val_acc=warmup_val_acc,
            )
        except Exception as e:
            print(f"  실패: {e}")
            traceback.print_exc()
            result = {
                "epoch_logs": [], "window_logs": [],
                "phi_snapshots": {}, "status": f"FAILED ({e})",
            }

        results[name] = result
        print(f"  [{name}] 완료 — 상태: {result['status']}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    generate_summary(results, warmup_log)

    # analysis.py용 결과 데이터 저장
    # (phi_snapshots에 큰 텐서가 있으므로 별도 저장)
    save_data = {}
    for name, result in results.items():
        save_data[name] = {
            "epoch_logs": result["epoch_logs"],
            "window_logs": result["window_logs"],
            "status": result["status"],
        }
        # Phi 스냅샷은 개별 파일로 저장
        for ep, phi_tensor in result.get("phi_snapshots", {}).items():
            torch.save(phi_tensor, f"results/phi_heatmaps/{name}_phi_epoch{ep}.pt")

    save_data["warmup_log"] = warmup_log
    torch.save(save_data, "results/train_ala_results.pt")
    print("결과 데이터 저장: results/train_ala_results.pt")

    print("\n모든 실험 완료.")


if __name__ == "__main__":
    main()
