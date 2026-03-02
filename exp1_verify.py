"""Exp1 (CE blend) standalone with policy verification logging.

Logs per K-window:
  1. Policy entropy (should start ~1.1, decrease over time)
  2. Confusion-conditioned actions (top-20 vs bottom-20 confused pairs)
  3. Phi movement stats (abs mean, max, clamp usage %)
  4. Reward moving average (last 10 windows)
  5. Confusion-Phi Pearson correlation

Based on ablation_fixes.py run_experiment(ce_blend=True, phi_penalty=False).
"""

import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import set_seed, get_dataloaders, get_model, get_optimizer_and_scheduler
from adaptive_loss import AdaptiveLoss
from state import construct_states, get_pair_indices_tensor
from controller import ALAPolicy, ReplayMemory, compute_reward
from train_ala import (
    fast_evaluate,
    fast_confusion_matrix,
    actions_to_delta_phi,
    update_policy,
    log_and_print,
)

# ── Constants ─────────────────────────────────────────────────────────────
WARMUP_EPOCHS = 50
TOTAL_EPOCHS = 200
NUM_CLASSES = 100
K = 200
BETA = 0.1
GAMMA = 0.9
POLICY_BATCH_SIZE = 8
PHI_SCALE = 0.01
CE_ALPHA = 0.5


# ── Verification helpers ──────────────────────────────────────────────────

def compute_policy_entropy(policy: ALAPolicy, states: torch.Tensor) -> float:
    """Mean entropy of policy distribution across all pairs."""
    with torch.no_grad():
        logits = policy(states)
        dist = Categorical(logits=logits)
        return dist.entropy().mean().item()


def confusion_conditioned_actions(
    actions: torch.Tensor,
    confusion: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    top_k: int = 20,
) -> dict:
    """Compare actions for top-K vs bottom-K confused pairs.

    Confusion score per pair = C[i,j] + C[j,i] (symmetric confusion).
    Action mapping: 0=-β, 1=0, 2=+β → mapped to -1, 0, +1 for analysis.
    """
    # Symmetric confusion score per pair
    conf_scores = confusion[pair_i, pair_j] + confusion[pair_j, pair_i]

    # Top-K (most confused) and Bottom-K (least confused)
    _, top_idx = conf_scores.topk(top_k, largest=True)
    _, bot_idx = conf_scores.topk(top_k, largest=False)

    def action_stats(idx):
        a = actions[idx]
        return {
            "plus": (a == 2).sum().item(),
            "zero": (a == 1).sum().item(),
            "minus": (a == 0).sum().item(),
        }

    return {
        "top": action_stats(top_idx),
        "bot": action_stats(bot_idx),
        "top_conf_mean": conf_scores[top_idx].mean().item(),
        "bot_conf_mean": conf_scores[bot_idx].mean().item(),
    }


def phi_movement_stats(phi: torch.Tensor, num_classes: int) -> dict:
    """Compute Phi off-diagonal movement from identity."""
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=phi.device)
    off_diag = phi[mask]
    abs_vals = off_diag.abs()
    clamp_limit = 0.1
    return {
        "abs_mean": abs_vals.mean().item(),
        "abs_max": abs_vals.max().item(),
        "usage_pct": (abs_vals.max().item() / clamp_limit) * 100,
        "nonzero_pct": (abs_vals > 1e-6).float().mean().item() * 100,
    }


def confusion_phi_correlation(
    confusion: torch.Tensor, phi: torch.Tensor, num_classes: int,
) -> float:
    """Pearson correlation between off-diagonal confusion and phi values."""
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=phi.device)
    c_flat = confusion[mask].float()
    p_flat = phi[mask].float()

    if c_flat.std() < 1e-8 or p_flat.std() < 1e-8:
        return 0.0

    c_centered = c_flat - c_flat.mean()
    p_centered = p_flat - p_flat.mean()
    corr = (c_centered * p_centered).sum() / (
        c_centered.norm() * p_centered.norm() + 1e-8
    )
    return corr.item()


# ── Warmup ────────────────────────────────────────────────────────────────

def run_warmup(train_loader, val_images, val_targets, test_images, test_targets,
               device, log_file):
    """Train 50 epochs with CE, save checkpoint."""
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, WARMUP_EPOCHS + 1):
        model.train()
        ep_loss, ep_total = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = ce(model(inputs), targets)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * inputs.size(0)
            ep_total += inputs.size(0)

        scheduler.step()
        val_acc = fast_evaluate(model, val_images, val_targets)
        test_acc = fast_evaluate(model, test_images, test_targets)
        log_and_print(
            f"Epoch {epoch:3d} | CE Warmup | Loss: {ep_loss/ep_total:.4f} | "
            f"Val: {val_acc:.2f}% | Test: {test_acc:.2f}%",
            log_file,
        )

    ckpt_path = "results/exp1_warmup.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_path)
    log_and_print(f"Warmup saved to {ckpt_path}", log_file)
    return ckpt_path


# ── Main experiment ───────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders()

    print("Pre-loading val/test data onto GPU...")
    val_images = torch.cat([x for x, _ in val_loader]).to(device)
    val_targets = torch.cat([y for _, y in val_loader]).to(device)
    test_images = torch.cat([x for x, _ in test_loader]).to(device)
    test_targets = torch.cat([y for _, y in test_loader]).to(device)

    os.makedirs("results", exist_ok=True)
    log_file = open("results/exp1_verify_log.txt", "w")

    # ── Warmup ────────────────────────────────────────────────────────
    log_and_print("=" * 70, log_file)
    log_and_print("Phase 0: CE Warmup (50 epochs)", log_file)
    log_and_print("=" * 70, log_file)
    ckpt_path = run_warmup(
        train_loader, val_images, val_targets,
        test_images, test_targets, device, log_file,
    )

    # ── Load checkpoint, init RL ──────────────────────────────────────
    log_and_print("=" * 70, log_file)
    log_and_print("Phase 1: CE Blend + ALA with verification logging", log_file)
    log_and_print("=" * 70, log_file)

    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    adaptive_loss = AdaptiveLoss(NUM_CLASSES).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    policy = ALAPolicy(24).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    memory = ReplayMemory(1000)
    pair_i, pair_j = get_pair_indices_tensor(NUM_CLASSES, device)

    steps_per_epoch = len(train_loader)
    total_steps = TOTAL_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    # State tracking
    confusion_history: list[torch.Tensor] = []
    M_old = None
    prev_states = None
    prev_actions = None
    prev_log_probs = None
    baseline_ema = 0.0
    reward_history = deque(maxlen=10)
    window_count = 0

    # Epoch tracking
    train_losses, val_accs, test_accs = [], [], []
    # Verification tracking (per window)
    verify_log = {
        "entropy": [], "reward_ma": [], "phi_abs_mean": [],
        "phi_max": [], "corr": [], "top_plus_pct": [], "bot_plus_pct": [],
    }

    global_step = warmup_steps
    current_epoch = WARMUP_EPOCHS
    epoch_loss, epoch_total = 0.0, 0
    train_iter = iter(train_loader)

    pbar = tqdm(total=total_steps - warmup_steps, desc="Exp1-Verify", unit="step")

    while global_step < total_steps:
        cumulative_metric = 0.0

        for j in range(1, K + 1):
            try:
                inputs, targets_batch = next(train_iter)
            except StopIteration:
                current_epoch += 1
                avg_loss = epoch_loss / max(epoch_total, 1)
                val_acc = fast_evaluate(model, val_images, val_targets)
                test_acc = fast_evaluate(model, test_images, test_targets)
                train_losses.append(avg_loss)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                log_and_print(
                    f"Epoch {current_epoch:3d} | Loss: {avg_loss:.4f} | "
                    f"Val: {val_acc:.2f}% | Test: {test_acc:.2f}%",
                    log_file,
                )

                scheduler.step()
                epoch_loss, epoch_total = 0.0, 0
                train_iter = iter(train_loader)
                inputs, targets_batch = next(train_iter)

            inputs, targets_batch = inputs.to(device), targets_batch.to(device)

            model.train()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = (CE_ALPHA * F.cross_entropy(logits, targets_batch)
                    + (1 - CE_ALPHA) * adaptive_loss(logits, targets_batch))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_total += inputs.size(0)
            global_step += 1
            pbar.update(1)

            # Eq.4: cumulative discounted metric
            val_acc_step = fast_evaluate(model, val_images, val_targets)
            val_error_step = 100.0 - val_acc_step
            cumulative_metric += GAMMA ** (K - j) * val_error_step
            model.train()

            if global_step >= total_steps:
                break

        # === End of K-window: RL update + verification ===
        window_count += 1
        M_new = cumulative_metric

        # Reward
        if M_old is not None and prev_states is not None:
            reward = compute_reward(M_old, M_new)
            memory.push(prev_states, prev_actions, prev_log_probs, reward)
            reward_history.append(reward)

        # Policy update
        if len(memory) >= POLICY_BATCH_SIZE:
            baseline_ema = update_policy(
                policy, policy_optimizer, memory,
                POLICY_BATCH_SIZE, baseline_ema, device,
            )

        # Confusion → state → action → Φ update
        C = fast_confusion_matrix(model, val_images, val_targets, NUM_CLASSES)
        confusion_history.append(C)
        if len(confusion_history) > 10:
            confusion_history.pop(0)

        progress = global_step / total_steps
        all_states = construct_states(
            confusion_history, adaptive_loss.phi.data, progress,
            NUM_CLASSES, pair_i, pair_j,
        )

        with torch.no_grad():
            actions, log_probs = policy.select_action(all_states)

        delta_phi = actions_to_delta_phi(
            actions, pair_i, pair_j, NUM_CLASSES, BETA, device,
        )
        adaptive_loss.update_phi(delta_phi * PHI_SCALE)

        # ──────────────────────────────────────────────────────────────
        # VERIFICATION LOGGING
        # ──────────────────────────────────────────────────────────────

        # 1. Policy entropy
        entropy = compute_policy_entropy(policy, all_states.to(device))

        # 2. Confusion-conditioned actions
        cca = confusion_conditioned_actions(actions, C, pair_i, pair_j, top_k=20)

        # 3. Phi movement
        phi_stats = phi_movement_stats(adaptive_loss.phi.data, NUM_CLASSES)

        # 4. Reward MA
        reward_ma = sum(reward_history) / len(reward_history) if reward_history else 0.0

        # 5. Confusion-Phi correlation
        corr = confusion_phi_correlation(C, adaptive_loss.phi.data, NUM_CLASSES)

        # Action distribution
        n_plus = (actions == 2).sum().item()
        n_zero = (actions == 1).sum().item()
        n_minus = (actions == 0).sum().item()
        n_total = actions.numel()

        # Top/bottom plus percentage
        top_total = cca["top"]["plus"] + cca["top"]["zero"] + cca["top"]["minus"]
        bot_total = cca["bot"]["plus"] + cca["bot"]["zero"] + cca["bot"]["minus"]
        top_plus_pct = cca["top"]["plus"] / max(top_total, 1) * 100
        bot_plus_pct = cca["bot"]["plus"] / max(bot_total, 1) * 100

        # Store for plotting
        verify_log["entropy"].append(entropy)
        verify_log["reward_ma"].append(reward_ma)
        verify_log["phi_abs_mean"].append(phi_stats["abs_mean"])
        verify_log["phi_max"].append(phi_stats["abs_max"])
        verify_log["corr"].append(corr)
        verify_log["top_plus_pct"].append(top_plus_pct)
        verify_log["bot_plus_pct"].append(bot_plus_pct)

        # Log
        log_and_print(
            f"  [Window {window_count:3d}] "
            f"Entropy: {entropy:.3f} | "
            f"Reward MA: {reward_ma:+.2f} | "
            f"Phi: abs={phi_stats['abs_mean']:.4f} max={phi_stats['abs_max']:.4f} "
            f"usage={phi_stats['usage_pct']:.1f}% | "
            f"Corr: {corr:+.3f}",
            log_file,
        )
        log_and_print(
            f"           "
            f"Actions: +={n_plus}({n_plus/n_total*100:.0f}%) "
            f"0={n_zero}({n_zero/n_total*100:.0f}%) "
            f"-={n_minus}({n_minus/n_total*100:.0f}%) | "
            f"Top20: +={cca['top']['plus']} 0={cca['top']['zero']} -={cca['top']['minus']} | "
            f"Bot20: +={cca['bot']['plus']} 0={cca['bot']['zero']} -={cca['bot']['minus']}",
            log_file,
        )

        M_old = M_new
        prev_states = all_states
        prev_actions = actions
        prev_log_probs = log_probs

    pbar.close()

    # Final epoch
    if epoch_total > 0:
        current_epoch += 1
        avg_loss = epoch_loss / epoch_total
        val_acc = fast_evaluate(model, val_images, val_targets)
        test_acc = fast_evaluate(model, test_images, test_targets)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        log_and_print(
            f"Epoch {current_epoch:3d} | Loss: {avg_loss:.4f} | "
            f"Val: {val_acc:.2f}% | Test: {test_acc:.2f}%",
            log_file,
        )

    # ── Summary ───────────────────────────────────────────────────────
    log_and_print("\n" + "=" * 70, log_file)
    log_and_print("VERIFICATION SUMMARY", log_file)
    log_and_print("=" * 70, log_file)

    ent = verify_log["entropy"]
    log_and_print(
        f"Entropy:  start={ent[0]:.3f} → end={ent[-1]:.3f} "
        f"(delta={ent[-1]-ent[0]:+.3f})", log_file,
    )
    log_and_print(
        f"Reward MA: final={verify_log['reward_ma'][-1]:+.2f}", log_file,
    )
    log_and_print(
        f"Phi movement: final abs_mean={verify_log['phi_abs_mean'][-1]:.4f} "
        f"max={verify_log['phi_max'][-1]:.4f}", log_file,
    )
    log_and_print(
        f"Confusion-Phi corr: final={verify_log['corr'][-1]:+.3f}", log_file,
    )

    tp = verify_log["top_plus_pct"]
    bp = verify_log["bot_plus_pct"]
    log_and_print(
        f"Top20 +action%: start={tp[0]:.0f}% → end={tp[-1]:.0f}% | "
        f"Bot20 +action%: start={bp[0]:.0f}% → end={bp[-1]:.0f}%", log_file,
    )

    if test_accs:
        log_and_print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%", log_file)

    log_file.close()

    # ── Verification plots ────────────────────────────────────────────
    os.makedirs("results/curves", exist_ok=True)
    windows = range(1, len(verify_log["entropy"]) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Exp1 CE Blend — Policy Verification", fontsize=14)

    # Entropy
    axes[0, 0].plot(windows, verify_log["entropy"])
    axes[0, 0].axhline(y=1.099, color="r", linestyle="--", alpha=0.5, label="max (log3)")
    axes[0, 0].set_ylabel("Policy Entropy")
    axes[0, 0].set_title("Policy Entropy over Windows")
    axes[0, 0].legend()

    # Reward MA
    axes[0, 1].plot(windows, verify_log["reward_ma"])
    axes[0, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].set_ylabel("Reward MA")
    axes[0, 1].set_title("Reward Moving Average (10-window)")

    # Phi movement
    axes[1, 0].plot(windows, verify_log["phi_abs_mean"], label="abs mean")
    axes[1, 0].plot(windows, verify_log["phi_max"], label="max")
    axes[1, 0].axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="clamp limit")
    axes[1, 0].set_ylabel("Phi off-diag")
    axes[1, 0].set_title("Phi Movement")
    axes[1, 0].legend()

    # Confusion-Phi correlation
    axes[1, 1].plot(windows, verify_log["corr"])
    axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].set_ylabel("Pearson r")
    axes[1, 1].set_title("Confusion–Phi Correlation")

    # Top vs Bottom +action %
    axes[2, 0].plot(windows, verify_log["top_plus_pct"], label="Top-20 confused")
    axes[2, 0].plot(windows, verify_log["bot_plus_pct"], label="Bottom-20 confused")
    axes[2, 0].set_ylabel("+action %")
    axes[2, 0].set_xlabel("K-window")
    axes[2, 0].set_title("+ Action Rate: High vs Low Confusion Pairs")
    axes[2, 0].legend()

    # Train loss / test acc
    if train_losses:
        ep_range = range(WARMUP_EPOCHS + 1, WARMUP_EPOCHS + 1 + len(test_accs))
        axes[2, 1].plot(ep_range, test_accs)
        axes[2, 1].set_ylabel("Test Accuracy (%)")
        axes[2, 1].set_xlabel("Epoch")
        axes[2, 1].set_title("Test Accuracy (post-warmup)")

    plt.tight_layout()
    plt.savefig("results/curves/exp1_verify.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to results/curves/exp1_verify.png")
    print(f"Log saved to results/exp1_verify_log.txt")


if __name__ == "__main__":
    main()
