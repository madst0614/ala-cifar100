"""Exp2: ALA Ablation Study — 9 experiments testing σ-scaling, Φ updates, reward types.

Phase 0: Warmup ResNet-18 with CE loss for 50 epochs, save checkpoint.
Phase 1: Run 9 experiments (1 baseline + 8 ALA variants) from warmup checkpoint.
Phase 2: Generate comparison table and plots.
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

from utils import set_seed, get_dataloaders, get_model, get_optimizer_and_scheduler
from adaptive_loss import AdaptiveLoss
from state import construct_states, get_pair_indices_tensor
from controller import ALAPolicy, ReplayMemory

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

CONFIGS = OrderedDict({
    "baseline_ce": {
        "description": "CE only (no ALA), 150 more epochs from warmup",
        "use_ala": False,
    },
    "exp_a": {
        "description": "σ(z / num_classes), clamp[-1,1], delta_scale=0.01",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": None,
        "reward_type": "sign",
    },
    "exp_b": {
        "description": "σ(z / sqrt(num_classes)), clamp[-1,1], delta_scale=0.01",
        "use_ala": True,
        "sigma_scale": "sqrt_num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": None,
        "reward_type": "sign",
    },
    "exp_c": {
        "description": "σ(z / running_mean_abs(z)), adaptive normalize, clamp[-1,1], delta_scale=0.01",
        "use_ala": True,
        "sigma_scale": "adaptive",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": None,
        "reward_type": "sign",
    },
    "exp_d": {
        "description": "σ(/num_classes), clamp[-1,1], delta_scale=0.1 (aggressive)",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.1,
        "selective_k": None,
        "reward_type": "sign",
    },
    "exp_e": {
        "description": "σ(/num_classes), clamp[-1,1], delta_scale=0.01, selective top-500 pairs",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": 500,
        "reward_type": "sign",
    },
    "exp_f": {
        "description": "σ(/num_classes), clamp[-1,1], delta_scale=0.01, continuous reward",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": None,
        "reward_type": "continuous",
    },
    "exp_g": {
        "description": "σ(/num_classes) + clamp[-1,1] + delta=0.01 + selective-500",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": 500,
        "reward_type": "sign",
    },
    "exp_h": {
        "description": "σ(/num_classes) + clamp[-1,1] + delta=0.01 + continuous reward + selective-500",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": 500,
        "reward_type": "continuous",
    },
})

NUM_CLASSES = 100
WARMUP_EPOCHS = 50
TOTAL_EPOCHS = 200
POST_WARMUP_EPOCHS = 150
BETA = 0.1
K = 200
GAMMA = 0.9
POLICY_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Modified AdaptiveLoss with sigma scaling support
# ---------------------------------------------------------------------------

class Exp2AdaptiveLoss(nn.Module):
    """AdaptiveLoss with configurable sigma scaling and clamp range."""

    def __init__(self, num_classes: int, sigma_scale: str, clamp_range: tuple) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sigma_scale = sigma_scale
        self.clamp_range = clamp_range
        self.phi = nn.Parameter(torch.eye(num_classes), requires_grad=False)
        # For adaptive scaling
        self.inner_ema = 0.0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        y = F.one_hot(targets, self.num_classes).float()
        weighted = y @ self.phi
        inner = (weighted * log_probs).sum(dim=1)

        # Apply sigma scaling
        if self.sigma_scale == "num_classes":
            inner = inner / self.num_classes
        elif self.sigma_scale == "sqrt_num_classes":
            inner = inner / math.sqrt(self.num_classes)
        elif self.sigma_scale == "adaptive":
            self.inner_ema = 0.99 * self.inner_ema + 0.01 * inner.detach().abs().mean().item()
            inner = inner / (self.inner_ema + 1e-8)

        sig = torch.sigmoid(inner)
        return (-sig).mean()

    def update_phi(self, delta_phi: torch.Tensor) -> None:
        self.phi.data += delta_phi
        self.phi.data.fill_diagonal_(1.0)
        self.phi.data = (self.phi.data + self.phi.data.T) / 2
        self.phi.data.fill_diagonal_(1.0)
        diag_mask = torch.eye(self.num_classes, device=self.phi.device, dtype=torch.bool)
        lo, hi = self.clamp_range
        self.phi.data[~diag_mask] = self.phi.data[~diag_mask].clamp(lo, hi)


# ---------------------------------------------------------------------------
# GPU-resident evaluation helpers (from train_ala.py pattern)
# ---------------------------------------------------------------------------

@torch.no_grad()
def fast_evaluate(model, images, targets, batch_size=2048):
    model.eval()
    correct = 0
    with torch.cuda.amp.autocast():
        for i in range(0, len(images), batch_size):
            logits = model(images[i:i + batch_size])
            correct += (logits.argmax(1) == targets[i:i + batch_size]).sum().item()
    return correct / len(images) * 100.0


@torch.no_grad()
def fast_confusion_matrix(model, images, targets, num_classes, batch_size=2048):
    model.eval()
    C = torch.zeros(num_classes, num_classes, device=images.device)
    class_counts = torch.zeros(num_classes, device=images.device)
    with torch.cuda.amp.autocast():
        for i in range(0, len(images), batch_size):
            logits = model(images[i:i + batch_size])
            neg_log_probs = -F.log_softmax(logits.float(), dim=1)
            bt = targets[i:i + batch_size]
            C.index_add_(0, bt, neg_log_probs)
            class_counts.scatter_add_(0, bt, torch.ones(bt.size(0), device=images.device))
    nonzero = class_counts > 0
    C[nonzero] /= class_counts[nonzero].unsqueeze(1)
    return C


# ---------------------------------------------------------------------------
# Helper: actions to delta_phi
# ---------------------------------------------------------------------------

def actions_to_delta_phi(actions, pair_i, pair_j, num_classes, beta, device):
    action_map = torch.tensor([-beta, 0.0, beta], device=device)
    delta_values = action_map[actions]
    delta_phi = torch.zeros(num_classes, num_classes, device=device)
    delta_phi[pair_i, pair_j] = delta_values
    delta_phi[pair_j, pair_i] = delta_values
    return delta_phi


# ---------------------------------------------------------------------------
# Helper: REINFORCE policy update
# ---------------------------------------------------------------------------

def update_policy(policy, optimizer, memory, batch_size, baseline_ema, device):
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
# Helper: compute reward
# ---------------------------------------------------------------------------

def compute_reward(M_old, M_new, reward_type):
    if reward_type == "sign":
        diff = M_old - M_new
        if diff > 0:
            return 1.0
        elif diff < 0:
            return -1.0
        return 0.0
    else:  # continuous
        return max(-1.0, min(1.0, (M_old - M_new) * 10.0))


# ---------------------------------------------------------------------------
# Helper: compute policy entropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_policy_entropy(policy, states):
    logits = policy(states)
    dist = Categorical(logits=logits)
    return dist.entropy().mean().item()


# ---------------------------------------------------------------------------
# Helper: confusion-phi Pearson correlation (off-diagonal)
# ---------------------------------------------------------------------------

def confusion_phi_correlation(confusion, phi, num_classes):
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
# Phase 0: Warmup
# ---------------------------------------------------------------------------

def run_warmup(device, train_loader, val_images, val_targets, test_images, test_targets):
    print("=" * 70)
    print("Phase 0: CE Warmup (50 epochs)")
    print("=" * 70)

    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, T_max=TOTAL_EPOCHS)
    ce_loss_fn = nn.CrossEntropyLoss()

    warmup_log = []
    best_val_acc = 0.0

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{WARMUP_EPOCHS} | "
                f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
            )

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": WARMUP_EPOCHS,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }
    ckpt_path = "results/exp2_warmup_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"\nWarmup complete. Val: {val_acc:.2f}%, Test: {test_acc:.2f}%")
    print(f"Checkpoint saved: {ckpt_path}")
    return ckpt_path, warmup_log


# ---------------------------------------------------------------------------
# Phase 1: Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    name, config, ckpt_path, device,
    train_loader, val_images, val_targets, test_images, test_targets,
):
    """Run one experiment from warmup checkpoint for POST_WARMUP_EPOCHS more epochs."""

    print(f"\n{'=' * 70}")
    print(f"Experiment: {name}")
    print(f"Description: {config['description']}")
    print(f"{'=' * 70}")

    use_ala = config["use_ala"]

    # Reload model, optimizer, scheduler from checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = get_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer, scheduler = get_optimizer_and_scheduler(model, T_max=TOTAL_EPOCHS)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    ce_loss_fn = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)

    # Per-epoch logs
    epoch_logs = []  # {epoch, train_loss, val_acc, test_acc}

    # Per-window logs (ALA only)
    window_logs = []  # {step, entropy, plus_ratio, zero_ratio, minus_ratio,
    #                    phi_abs_mean, phi_abs_max, corr, reward}

    pair_i, pair_j = get_pair_indices_tensor(NUM_CLASSES, device)
    num_pairs = NUM_CLASSES * (NUM_CLASSES - 1) // 2  # 4950

    if not use_ala:
        # === CE Baseline: simple epoch loop ===
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

        return {"epoch_logs": epoch_logs, "window_logs": window_logs, "status": "OK"}

    # === ALA Experiment ===
    sigma_scale = config["sigma_scale"]
    clamp_range = config["clamp_range"]
    delta_scale = config["delta_scale"]
    selective_k = config["selective_k"]
    reward_type = config["reward_type"]

    adaptive_loss = Exp2AdaptiveLoss(NUM_CLASSES, sigma_scale, clamp_range).to(device)
    policy = ALAPolicy(state_dim=24).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    memory = ReplayMemory(1000)

    confusion_history = []
    M_old = None
    prev_states = None
    prev_actions = None
    prev_log_probs = None
    baseline_ema = 0.0

    # K-window training loop
    global_step = 0
    total_steps = POST_WARMUP_EPOCHS * steps_per_epoch
    current_epoch = WARMUP_EPOCHS
    epoch_loss = 0.0
    epoch_total = 0
    train_iter = iter(train_loader)

    # Phi snapshot epochs (real epochs)
    phi_snapshot_epochs = {WARMUP_EPOCHS + 1, 100, 150, 200}

    pbar = tqdm(total=total_steps, desc=f"  {name}", unit="step", leave=False)

    while global_step < total_steps:
        # === One K-step window ===
        window_start_error = 100.0 - fast_evaluate(model, val_images, val_targets)

        for j in range(1, K + 1):
            # Get next batch
            try:
                inputs, targets_batch = next(train_iter)
            except StopIteration:
                # Epoch boundary
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

                ep_rel = current_epoch - WARMUP_EPOCHS
                if ep_rel % 10 == 0 or ep_rel == 1:
                    print(
                        f"  Epoch {current_epoch:3d}/{TOTAL_EPOCHS} | "
                        f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
                    )

                # Phi snapshots
                if current_epoch in phi_snapshot_epochs:
                    torch.save(
                        adaptive_loss.phi.data.cpu(),
                        f"results/exp2_{name}_phi_{current_epoch}.pt",
                    )

                scheduler.step()
                epoch_loss = 0.0
                epoch_total = 0
                train_iter = iter(train_loader)
                inputs, targets_batch = next(train_iter)

            inputs, targets_batch = inputs.to(device), targets_batch.to(device)

            # SGD step with adaptive loss
            model.train()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = adaptive_loss(logits, targets_batch)

            # NaN check
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at step {global_step}, aborting experiment.")
                pbar.close()
                return {"epoch_logs": epoch_logs, "window_logs": window_logs, "status": "FAILED (NaN)"}

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_total += inputs.size(0)
            global_step += 1
            pbar.update(1)

            if global_step >= total_steps:
                break

        # === End of K-step window ===
        window_end_error = 100.0 - fast_evaluate(model, val_images, val_targets)

        M_new = window_end_error

        # Reward
        reward_val = None
        if M_old is not None and prev_states is not None:
            reward_val = compute_reward(M_old, M_new, reward_type)
            memory.push(prev_states, prev_actions, prev_log_probs, reward_val)

        # Policy update
        if len(memory) >= POLICY_BATCH_SIZE:
            baseline_ema = update_policy(
                policy, policy_optimizer, memory,
                POLICY_BATCH_SIZE, baseline_ema, device,
            )

        # Confusion matrix → state → action → Φ update
        progress = global_step / total_steps
        C = fast_confusion_matrix(model, val_images, val_targets, NUM_CLASSES)
        confusion_history.append(C)
        if len(confusion_history) > 10:
            confusion_history.pop(0)

        all_states = construct_states(
            confusion_history, adaptive_loss.phi.data, progress,
            NUM_CLASSES, pair_i, pair_j,
        )

        # Selective-k: determine which pairs get actions
        if selective_k is not None:
            # Confusion score per pair: C[i,j] + C[j,i]
            pair_scores = C[pair_i, pair_j] + C[pair_j, pair_i]
            _, topk_indices = torch.topk(pair_scores, selective_k)
            topk_mask = torch.zeros(num_pairs, dtype=torch.bool, device=device)
            topk_mask[topk_indices] = True

            # Sample actions only for top-k pairs
            topk_states = all_states[topk_mask]
            with torch.no_grad():
                topk_actions, topk_log_probs = policy.select_action(topk_states)

            # Full action tensor: index 1 = zero action for non-selected pairs
            actions = torch.ones(num_pairs, dtype=torch.long, device=device)
            log_probs = torch.zeros(num_pairs, device=device)
            actions[topk_mask] = topk_actions
            log_probs[topk_mask] = topk_log_probs
        else:
            with torch.no_grad():
                actions, log_probs = policy.select_action(all_states)

        delta_phi = actions_to_delta_phi(
            actions, pair_i, pair_j, NUM_CLASSES, BETA, device,
        )
        adaptive_loss.update_phi(delta_phi * delta_scale)

        # === Logging ===
        # Policy entropy (over all states, or topk if selective)
        entropy = compute_policy_entropy(policy, all_states)

        # Action distribution
        n_plus = (actions == 2).sum().item()
        n_zero = (actions == 1).sum().item()
        n_minus = (actions == 0).sum().item()
        total_a = actions.numel()

        # Phi stats (off-diagonal)
        diag_mask = torch.eye(NUM_CLASSES, dtype=torch.bool, device=device)
        phi_off = adaptive_loss.phi.data[~diag_mask]
        phi_abs_mean = phi_off.abs().mean().item()
        phi_abs_max = phi_off.abs().max().item()

        # Confusion-Phi correlation
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

    # Final epoch if pending
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
        if current_epoch in phi_snapshot_epochs:
            torch.save(
                adaptive_loss.phi.data.cpu(),
                f"results/exp2_{name}_phi_{current_epoch}.pt",
            )

    return {"epoch_logs": epoch_logs, "window_logs": window_logs, "status": "OK"}


# ---------------------------------------------------------------------------
# Phase 2: Results
# ---------------------------------------------------------------------------

def save_individual_log(name, result):
    path = f"results/exp2_{name}_log.txt"
    with open(path, "w") as f:
        f.write(f"Experiment: {name}\n")
        f.write(f"Status: {result['status']}\n\n")

        f.write("=== Epoch Logs ===\n")
        f.write(f"{'Epoch':>6} {'TrainLoss':>10} {'ValAcc':>8} {'TestAcc':>8}\n")
        for e in result["epoch_logs"]:
            f.write(
                f"{e['epoch']:6d} {e['train_loss']:10.4f} "
                f"{e['val_acc']:8.2f} {e['test_acc']:8.2f}\n"
            )

        if result["window_logs"]:
            f.write("\n=== Window Logs (RL) ===\n")
            f.write(
                f"{'Step':>8} {'Entropy':>8} {'Plus%':>7} {'Zero%':>7} {'Minus%':>7} "
                f"{'PhiAbsMn':>9} {'PhiAbsMx':>9} {'Corr':>7} {'Reward':>7}\n"
            )
            for w in result["window_logs"]:
                f.write(
                    f"{w['step']:8d} {w['entropy']:8.4f} "
                    f"{w['plus_ratio']*100:7.1f} {w['zero_ratio']*100:7.1f} "
                    f"{w['minus_ratio']*100:7.1f} "
                    f"{w['phi_abs_mean']:9.6f} {w['phi_abs_max']:9.6f} "
                    f"{w['corr']:+7.4f} {w['reward']:+7.3f}\n"
                )


def generate_summary(results):
    lines = []
    header = (
        f"{'Experiment':<17} | {'Final Test Acc':>14} | {'Best Test Acc':>13} | "
        f"{'Final Entropy':>13} | {'Final Phi abs_mean':>18} | {'Final Corr':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name, result in results.items():
        if result["status"] != "OK":
            lines.append(f"{name:<17} | {'FAILED':>14} | {'FAILED':>13} | "
                         f"{'-':>13} | {'-':>18} | {'-':>10}")
            continue

        elogs = result["epoch_logs"]
        if not elogs:
            lines.append(f"{name:<17} | {'NO DATA':>14} | {'NO DATA':>13} | "
                         f"{'-':>13} | {'-':>18} | {'-':>10}")
            continue

        final_test = elogs[-1]["test_acc"]
        best_test = max(e["test_acc"] for e in elogs)

        wlogs = result["window_logs"]
        if wlogs:
            final_entropy = f"{wlogs[-1]['entropy']:.3f}"
            final_phi = f"{wlogs[-1]['phi_abs_mean']:.6f}"
            final_corr = f"{wlogs[-1]['corr']:+.4f}"
        else:
            final_entropy = "-"
            final_phi = "-"
            final_corr = "-"

        lines.append(
            f"{name:<17} | {final_test:13.2f}% | {best_test:12.2f}% | "
            f"{final_entropy:>13} | {final_phi:>18} | {final_corr:>10}"
        )

    text = "\n".join(lines)
    print("\n" + text)
    with open("results/exp2_summary.txt", "w") as f:
        f.write(text + "\n")
    print("\nSummary saved to results/exp2_summary.txt")


def generate_plots(results, warmup_log):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Build warmup epoch range for x-axis continuity
    warmup_epochs_x = [e["epoch"] for e in warmup_log]
    warmup_test_accs = [e["test_acc"] for e in warmup_log]

    # Subplot 1: Test accuracy curves
    ax = axes[0, 0]
    ax.plot(warmup_epochs_x, warmup_test_accs, "k--", alpha=0.3, label="_warmup")
    for name, result in results.items():
        if result["status"] != "OK" or not result["epoch_logs"]:
            continue
        epochs = [e["epoch"] for e in result["epoch_logs"]]
        test_accs = [e["test_acc"] for e in result["epoch_logs"]]
        ax.plot(epochs, test_accs, label=name, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Policy entropy
    ax = axes[0, 1]
    for name, result in results.items():
        if not result["window_logs"]:
            continue
        steps = [w["step"] for w in result["window_logs"]]
        entropy = [w["entropy"] for w in result["window_logs"]]
        ax.plot(steps, entropy, label=name, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy (ALA only)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Phi abs_mean
    ax = axes[1, 0]
    for name, result in results.items():
        if not result["window_logs"]:
            continue
        steps = [w["step"] for w in result["window_logs"]]
        phi_am = [w["phi_abs_mean"] for w in result["window_logs"]]
        ax.plot(steps, phi_am, label=name, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Phi Off-Diag |mean|")
    ax.set_title("Φ abs_mean")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 4: Confusion-Phi correlation
    ax = axes[1, 1]
    for name, result in results.items():
        if not result["window_logs"]:
            continue
        steps = [w["step"] for w in result["window_logs"]]
        corr = [w["corr"] for w in result["window_logs"]]
        ax.plot(steps, corr, label=name, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Confusion–Φ Correlation (off-diagonal)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/exp2_comparison.png", dpi=150)
    plt.close()
    print("Comparison plot saved to results/exp2_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("results", exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # GPU preload val/test
    print("Pre-loading val/test data onto GPU...")
    val_images = torch.cat([x for x, _ in val_loader]).to(device)
    val_targets = torch.cat([y for _, y in val_loader]).to(device)
    test_images = torch.cat([x for x, _ in test_loader]).to(device)
    test_targets = torch.cat([y for _, y in test_loader]).to(device)
    print(f"  val: {val_images.shape}, test: {test_images.shape}")

    # Phase 0: Warmup
    ckpt_path = "results/exp2_warmup_checkpoint.pt"
    if os.path.exists(ckpt_path):
        print(f"\nWarmup checkpoint found at {ckpt_path}, skipping warmup.")
        warmup_log = []  # No warmup log available from cache
    else:
        ckpt_path, warmup_log = run_warmup(
            device, train_loader, val_images, val_targets, test_images, test_targets,
        )

    # Phase 1: Run experiments
    results = OrderedDict()
    total_exps = len(CONFIGS)
    for idx, (name, config) in enumerate(CONFIGS.items(), 1):
        print(f"\n[{idx}/{total_exps}] Starting experiment: {name}")
        try:
            result = run_experiment(
                name, config, ckpt_path, device,
                train_loader, val_images, val_targets, test_images, test_targets,
            )
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            traceback.print_exc()
            result = {"epoch_logs": [], "window_logs": [], "status": f"FAILED ({e})"}

        results[name] = result
        save_individual_log(name, result)
        print(f"  [{name}] Done — Status: {result['status']}")

    # Phase 2: Summary and plots
    print("\n" + "=" * 70)
    print("Phase 2: Results Summary")
    print("=" * 70)

    generate_summary(results)
    generate_plots(results, warmup_log)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
