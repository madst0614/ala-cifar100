"""Exp2: ALA Ablation Study — Warmup once, then fork 9 experiments.

Usage: python exp2.py

Phase 0: CE warmup 50 epochs → save checkpoint
Phase 1: 9 experiments from checkpoint (150 epochs each)
Phase 2: Comparison table + plots
"""

import copy
import math
import os
import traceback
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    set_seed,
    get_dataloaders,
    get_model,
    get_optimizer_and_scheduler,
)
from state import construct_states, get_pair_indices_tensor
from controller import ALAPolicy, ReplayMemory

# ═══════════════════════════════════════════════════════════════════════════
# Configs
# ═══════════════════════════════════════════════════════════════════════════

WARMUP_EPOCHS = 50
TOTAL_EPOCHS = 200
POST_WARMUP_EPOCHS = TOTAL_EPOCHS - WARMUP_EPOCHS  # 150
NUM_CLASSES = 100
K = 200
BETA = 0.1
GAMMA_DISCOUNT = 0.9
POLICY_BATCH_SIZE = 8
REPLAY_CAPACITY = 1000
SEED = 42

CONFIGS = {
    "baseline_ce": {
        "desc": "CE only (no ALA), 150 more epochs from warmup",
        "use_ala": False,
    },
    "exp_a": {
        "desc": "sigma(z/num_classes), clamp[-1,1], delta=0.01",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": None,
        "reward_type": "sign",
    },
    "exp_b": {
        "desc": "sigma(z/sqrt(num_classes)), clamp[-1,1], delta=0.01",
        "use_ala": True,
        "sigma_scale": "sqrt_num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": None,
        "reward_type": "sign",
    },
    "exp_c": {
        "desc": "sigma(z/adaptive_norm), clamp[-1,1], delta=0.01",
        "use_ala": True,
        "sigma_scale": "adaptive",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": None,
        "reward_type": "sign",
    },
    "exp_d": {
        "desc": "sigma(z/num_classes), clamp[-1,1], delta=0.1 (aggressive)",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.1,
        "selective_k": None,
        "reward_type": "sign",
    },
    "exp_e": {
        "desc": "sigma(z/num_classes), clamp[-1,1], delta=0.01, selective top-500",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": 500,
        "reward_type": "sign",
    },
    "exp_f": {
        "desc": "sigma(z/num_classes), clamp[-1,1], delta=0.01, continuous reward",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": None,
        "reward_type": "continuous",
    },
    "exp_g": {
        "desc": "sigma(z/num_classes) + delta=0.01 + selective-500 (combo)",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": 500,
        "reward_type": "sign",
    },
    "exp_h": {
        "desc": "sigma(z/num_classes) + delta=0.01 + selective-500 + continuous reward",
        "use_ala": True,
        "sigma_scale": "num_classes",
        "clamp_range": (-1.0, 1.0),
        "delta_scale": 0.01,
        "selective_k": 500,
        "reward_type": "continuous",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Configurable AdaptiveLoss
# ═══════════════════════════════════════════════════════════════════════════

class ConfigurableAdaptiveLoss(nn.Module):
    """AdaptiveLoss with configurable sigma scaling and clamp range."""

    def __init__(
        self,
        num_classes: int = 100,
        sigma_scale: str = "num_classes",
        clamp_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.sigma_scale = sigma_scale
        self.clamp_range = clamp_range
        self.phi = nn.Parameter(torch.eye(num_classes), requires_grad=False)
        # For adaptive scaling
        self.register_buffer("inner_ema", torch.tensor(1.0))

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
            with torch.no_grad():
                batch_abs_mean = inner.detach().abs().mean()
                self.inner_ema = 0.99 * self.inner_ema + 0.01 * batch_abs_mean
            inner = inner / (self.inner_ema + 1e-8)

        return (-torch.sigmoid(inner)).mean()

    def update_phi(self, delta_phi: torch.Tensor) -> None:
        self.phi.data += delta_phi
        self.phi.data.fill_diagonal_(1.0)
        self.phi.data = (self.phi.data + self.phi.data.T) / 2
        self.phi.data.fill_diagonal_(1.0)
        diag_mask = torch.eye(self.num_classes, device=self.phi.device, dtype=torch.bool)
        lo, hi = self.clamp_range
        self.phi.data[~diag_mask] = self.phi.data[~diag_mask].clamp(lo, hi)


# ═══════════════════════════════════════════════════════════════════════════
# Fast GPU-resident helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def fast_evaluate(model, images, targets, batch_size=2048):
    model.eval()
    correct = 0
    for i in range(0, len(images), batch_size):
        logits = model(images[i:i + batch_size])
        correct += (logits.argmax(1) == targets[i:i + batch_size]).sum().item()
    return correct / len(images) * 100.0


@torch.no_grad()
def fast_confusion_matrix(model, images, targets, num_classes, batch_size=2048):
    model.eval()
    C = torch.zeros(num_classes, num_classes, device=images.device)
    counts = torch.zeros(num_classes, device=images.device)
    for i in range(0, len(images), batch_size):
        logits = model(images[i:i + batch_size])
        nlp = -F.log_softmax(logits.float(), dim=1)
        bt = targets[i:i + batch_size]
        C.index_add_(0, bt, nlp)
        counts.scatter_add_(0, bt, torch.ones(bt.size(0), device=images.device))
    nonzero = counts > 0
    C[nonzero] /= counts[nonzero].unsqueeze(1)
    return C


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def actions_to_delta_phi(actions, pair_i, pair_j, num_classes, beta, device):
    action_map = torch.tensor([-beta, 0.0, beta], device=device)
    delta_values = action_map[actions]
    delta_phi = torch.zeros(num_classes, num_classes, device=device)
    delta_phi[pair_i, pair_j] = delta_values
    delta_phi[pair_j, pair_i] = delta_values
    return delta_phi


def compute_reward_sign(M_old, M_new):
    diff = M_old - M_new
    if diff > 0:
        return 1.0
    elif diff < 0:
        return -1.0
    return 0.0


def compute_reward_continuous(M_old, M_new, scale=10.0):
    diff = (M_old - M_new) * scale
    return max(-1.0, min(1.0, diff))


def update_policy(policy, optimizer, memory, batch_size, baseline_ema, device):
    samples = memory.sample(batch_size)
    total_loss = torch.tensor(0.0, device=device)
    total_reward = 0.0

    for states, actions, _old_lp, reward in samples:
        states = states.to(device)
        actions = actions.to(device)
        logits = policy(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        advantage = reward - baseline_ema
        total_loss = total_loss - (log_probs * advantage).mean()
        total_reward += reward

    total_loss = total_loss / len(samples)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    avg_reward = total_reward / len(samples)
    return 0.9 * baseline_ema + 0.1 * avg_reward


def policy_entropy(policy, states):
    with torch.no_grad():
        logits = policy(states)
        return Categorical(logits=logits).entropy().mean().item()


def confusion_phi_corr(confusion, phi, num_classes, device):
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=device)
    c = confusion[mask].float()
    p = phi[mask].float()
    if c.std() < 1e-8 or p.std() < 1e-8:
        return 0.0
    c = c - c.mean()
    p = p - p.mean()
    return (c * p).sum().item() / (c.norm().item() * p.norm().item() + 1e-8)


def log(msg, log_file, also_print=True):
    if also_print:
        tqdm.write(msg)
    log_file.write(msg + "\n")
    log_file.flush()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Warmup
# ═══════════════════════════════════════════════════════════════════════════

def run_warmup(train_loader, val_imgs, val_tgts, test_imgs, test_tgts, device):
    print("=" * 70)
    print("Phase 0: CE Warmup (50 epochs)")
    print("=" * 70)

    set_seed(SEED)
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, WARMUP_EPOCHS + 1):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * x.size(0)
            ep_n += x.size(0)
        scheduler.step()

        if epoch % 10 == 0 or epoch == WARMUP_EPOCHS:
            val_acc = fast_evaluate(model, val_imgs, val_tgts)
            test_acc = fast_evaluate(model, test_imgs, test_tgts)
            print(
                f"  Epoch {epoch:3d} | Loss: {ep_loss/ep_n:.4f} | "
                f"Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
            )

    ckpt_path = "results/exp2_warmup.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_path)
    print(f"  Warmup checkpoint saved: {ckpt_path}")

    val_acc = fast_evaluate(model, val_imgs, val_tgts)
    test_acc = fast_evaluate(model, test_imgs, test_tgts)
    print(f"  Warmup final — Val: {val_acc:.2f}% | Test: {test_acc:.2f}%")

    return ckpt_path, val_acc, test_acc


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Single experiment runner
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    name, cfg, ckpt_path,
    train_loader, val_imgs, val_tgts, test_imgs, test_tgts,
    device,
):
    """Run one experiment for POST_WARMUP_EPOCHS epochs from warmup checkpoint.

    Returns dict with curves and final metrics, or None on failure.
    """
    log_path = f"results/exp2_{name}_log.txt"
    lf = open(log_path, "w")
    log(f"Experiment: {name}", lf)
    log(f"Config: {cfg['desc']}", lf)
    log("=" * 60, lf)

    try:
        return _run_experiment_inner(
            name, cfg, ckpt_path,
            train_loader, val_imgs, val_tgts, test_imgs, test_tgts,
            device, lf,
        )
    except Exception as e:
        log(f"FAILED: {e}", lf)
        log(traceback.format_exc(), lf)
        return None
    finally:
        lf.close()


def _run_experiment_inner(
    name, cfg, ckpt_path,
    train_loader, val_imgs, val_tgts, test_imgs, test_tgts,
    device, lf,
):
    # ── Load checkpoint ──────────────────────────────────────────────
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    use_ala = cfg.get("use_ala", False)

    # ── CE-only baseline ─────────────────────────────────────────────
    if not use_ala:
        return _run_ce_baseline(
            name, model, optimizer, scheduler,
            train_loader, val_imgs, val_tgts, test_imgs, test_tgts,
            device, lf,
        )

    # ── ALA experiment ───────────────────────────────────────────────
    sigma_scale = cfg["sigma_scale"]
    clamp_range = cfg["clamp_range"]
    delta_scale = cfg["delta_scale"]
    selective_k = cfg.get("selective_k")
    reward_type = cfg.get("reward_type", "sign")

    adaptive_loss = ConfigurableAdaptiveLoss(
        NUM_CLASSES, sigma_scale, clamp_range,
    ).to(device)
    ce_criterion = nn.CrossEntropyLoss()

    policy = ALAPolicy(state_dim=24).to(device)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=0.001)
    memory = ReplayMemory(REPLAY_CAPACITY)

    pair_i, pair_j = get_pair_indices_tensor(NUM_CLASSES, device)
    num_pairs = pair_i.size(0)
    steps_per_epoch = len(train_loader)
    total_steps = POST_WARMUP_EPOCHS * steps_per_epoch

    # Tracking
    confusion_history: list[torch.Tensor] = []
    M_old = None
    prev_states = None
    prev_actions = None
    prev_log_probs = None
    baseline_ema = 0.0

    # Curves
    train_losses, val_accs, test_accs = [], [], []
    rl_log = {
        "entropy": [], "phi_abs_mean": [], "phi_abs_max": [],
        "corr": [], "reward": [],
        "act_plus": [], "act_zero": [], "act_minus": [],
    }

    global_step = 0
    current_epoch = 0
    epoch_loss, epoch_n = 0.0, 0
    train_iter = iter(train_loader)
    window_count = 0

    pbar = tqdm(total=total_steps, desc=f"  {name}", unit="step", leave=False)

    while global_step < total_steps:
        # ── One K-window ─────────────────────────────────────────────
        for j in range(1, K + 1):
            # Next batch (handle epoch boundary)
            try:
                inputs, targets_batch = next(train_iter)
            except StopIteration:
                current_epoch += 1
                avg_loss = epoch_loss / max(epoch_n, 1)
                va = fast_evaluate(model, val_imgs, val_tgts)
                ta = fast_evaluate(model, test_imgs, test_tgts)
                train_losses.append(avg_loss)
                val_accs.append(va)
                test_accs.append(ta)

                log(
                    f"  Epoch {WARMUP_EPOCHS + current_epoch:3d} | "
                    f"Loss: {avg_loss:.4f} | Val: {va:.2f}% | Test: {ta:.2f}%",
                    lf,
                )

                # Save Phi snapshots
                real_epoch = WARMUP_EPOCHS + current_epoch
                if real_epoch in (50, 100, 150, 200):
                    torch.save(
                        adaptive_loss.phi.data.cpu(),
                        f"results/exp2_{name}_phi_{real_epoch:03d}.pt",
                    )

                scheduler.step()
                epoch_loss, epoch_n = 0.0, 0
                train_iter = iter(train_loader)
                inputs, targets_batch = next(train_iter)

            inputs, targets_batch = inputs.to(device), targets_batch.to(device)

            # SGD step with adaptive loss
            model.train()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = adaptive_loss(logits, targets_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_n += inputs.size(0)
            global_step += 1
            pbar.update(1)

            if global_step >= total_steps:
                break

        # ── End of K-window: RL update ───────────────────────────────
        window_count += 1

        # Measure val error at window end
        val_error_now = 100.0 - fast_evaluate(model, val_imgs, val_tgts)
        M_new = val_error_now

        # Reward
        reward = 0.0
        if M_old is not None and prev_states is not None:
            if reward_type == "continuous":
                reward = compute_reward_continuous(M_old, M_new)
            else:
                reward = compute_reward_sign(M_old, M_new)
            memory.push(prev_states, prev_actions, prev_log_probs, reward)

        # Policy update
        if len(memory) >= POLICY_BATCH_SIZE:
            baseline_ema = update_policy(
                policy, policy_opt, memory,
                POLICY_BATCH_SIZE, baseline_ema, device,
            )

        # Confusion → state → action → Φ
        progress = global_step / total_steps
        C = fast_confusion_matrix(model, val_imgs, val_tgts, NUM_CLASSES)
        confusion_history.append(C)
        if len(confusion_history) > 10:
            confusion_history.pop(0)

        # Selective: pick top-k confused pairs
        if selective_k is not None and selective_k < num_pairs:
            conf_scores = C[pair_i, pair_j] + C[pair_j, pair_i]
            _, topk_idx = conf_scores.topk(selective_k, largest=True)
            sel_pair_i = pair_i[topk_idx]
            sel_pair_j = pair_j[topk_idx]

            sel_states = construct_states(
                confusion_history, adaptive_loss.phi.data, progress,
                NUM_CLASSES, sel_pair_i, sel_pair_j,
            )
            with torch.no_grad():
                sel_actions, sel_log_probs = policy.select_action(sel_states)

            # Full action tensor: default zero (action index 1)
            all_actions = torch.ones(num_pairs, dtype=torch.long, device=device)
            all_actions[topk_idx] = sel_actions
            all_log_probs = torch.zeros(num_pairs, device=device)
            all_log_probs[topk_idx] = sel_log_probs

            # For replay: store only selected states
            prev_states = sel_states
            prev_actions = sel_actions
            prev_log_probs = sel_log_probs

            delta_phi = actions_to_delta_phi(
                all_actions, pair_i, pair_j, NUM_CLASSES, BETA, device,
            )
        else:
            all_states = construct_states(
                confusion_history, adaptive_loss.phi.data, progress,
                NUM_CLASSES, pair_i, pair_j,
            )
            with torch.no_grad():
                all_actions, all_log_probs = policy.select_action(all_states)

            prev_states = all_states
            prev_actions = all_actions
            prev_log_probs = all_log_probs

            delta_phi = actions_to_delta_phi(
                all_actions, pair_i, pair_j, NUM_CLASSES, BETA, device,
            )

        adaptive_loss.update_phi(delta_phi * delta_scale)
        M_old = M_new

        # ── RL diagnostics ───────────────────────────────────────────
        # Entropy
        diag_states = (
            prev_states if selective_k is None
            else sel_states
        )
        ent = policy_entropy(policy, diag_states.to(device))

        # Phi stats
        dmask = ~torch.eye(NUM_CLASSES, dtype=torch.bool, device=device)
        off = adaptive_loss.phi.data[dmask]
        phi_am = off.abs().mean().item()
        phi_ax = off.abs().max().item()

        # Correlation
        corr = confusion_phi_corr(C, adaptive_loss.phi.data, NUM_CLASSES, device)

        # Action dist (from the actions actually applied)
        n_plus = (all_actions == 2).sum().item()
        n_zero = (all_actions == 1).sum().item()
        n_minus = (all_actions == 0).sum().item()
        n_total = all_actions.numel()

        rl_log["entropy"].append(ent)
        rl_log["phi_abs_mean"].append(phi_am)
        rl_log["phi_abs_max"].append(phi_ax)
        rl_log["corr"].append(corr)
        rl_log["reward"].append(reward)
        rl_log["act_plus"].append(n_plus / n_total * 100)
        rl_log["act_zero"].append(n_zero / n_total * 100)
        rl_log["act_minus"].append(n_minus / n_total * 100)

        if window_count % 20 == 0 or window_count <= 3:
            log(
                f"    [W{window_count:3d}] Ent:{ent:.3f} "
                f"Reward:{reward:+.1f} "
                f"Phi:{phi_am:.4f}/{phi_ax:.4f} "
                f"Corr:{corr:+.3f} "
                f"Act: +{n_plus/n_total*100:.0f}% 0={n_zero/n_total*100:.0f}% -{n_minus/n_total*100:.0f}%",
                lf,
            )

    pbar.close()

    # Final epoch if pending
    if epoch_n > 0:
        current_epoch += 1
        avg_loss = epoch_loss / epoch_n
        va = fast_evaluate(model, val_imgs, val_tgts)
        ta = fast_evaluate(model, test_imgs, test_tgts)
        train_losses.append(avg_loss)
        val_accs.append(va)
        test_accs.append(ta)
        real_epoch = WARMUP_EPOCHS + current_epoch
        if real_epoch in (50, 100, 150, 200):
            torch.save(
                adaptive_loss.phi.data.cpu(),
                f"results/exp2_{name}_phi_{real_epoch:03d}.pt",
            )

    final_test = test_accs[-1] if test_accs else 0.0
    best_test = max(test_accs) if test_accs else 0.0
    final_ent = rl_log["entropy"][-1] if rl_log["entropy"] else 0.0
    final_phi = rl_log["phi_abs_mean"][-1] if rl_log["phi_abs_mean"] else 0.0
    final_corr = rl_log["corr"][-1] if rl_log["corr"] else 0.0

    log(f"\n  Final Test: {final_test:.2f}% | Best Test: {best_test:.2f}%", lf)
    log(f"  Final Entropy: {final_ent:.3f} | Phi abs_mean: {final_phi:.4f} | Corr: {final_corr:+.3f}", lf)

    return {
        "train_losses": train_losses,
        "val_accs": val_accs,
        "test_accs": test_accs,
        "rl_log": rl_log,
        "final_test": final_test,
        "best_test": best_test,
        "final_entropy": final_ent,
        "final_phi_abs_mean": final_phi,
        "final_corr": final_corr,
    }


def _run_ce_baseline(
    name, model, optimizer, scheduler,
    train_loader, val_imgs, val_tgts, test_imgs, test_tgts,
    device, lf,
):
    """CE-only continuation from warmup checkpoint."""
    ce = nn.CrossEntropyLoss()
    train_losses, val_accs, test_accs = [], [], []

    for epoch in range(1, POST_WARMUP_EPOCHS + 1):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * x.size(0)
            ep_n += x.size(0)

        scheduler.step()
        avg_loss = ep_loss / ep_n
        va = fast_evaluate(model, val_imgs, val_tgts)
        ta = fast_evaluate(model, test_imgs, test_tgts)
        train_losses.append(avg_loss)
        val_accs.append(va)
        test_accs.append(ta)

        if epoch % 10 == 0 or epoch == POST_WARMUP_EPOCHS:
            log(
                f"  Epoch {WARMUP_EPOCHS + epoch:3d} | "
                f"Loss: {avg_loss:.4f} | Val: {va:.2f}% | Test: {ta:.2f}%",
                lf,
            )

    final_test = test_accs[-1] if test_accs else 0.0
    best_test = max(test_accs) if test_accs else 0.0
    log(f"\n  Final Test: {final_test:.2f}% | Best Test: {best_test:.2f}%", lf)

    return {
        "train_losses": train_losses,
        "val_accs": val_accs,
        "test_accs": test_accs,
        "rl_log": None,
        "final_test": final_test,
        "best_test": best_test,
        "final_entropy": None,
        "final_phi_abs_mean": None,
        "final_corr": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Comparison
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison(results, warmup_val, warmup_test):
    header = (
        f"{'Experiment':<16} | {'Final Test':>10} | {'Best Test':>10} | "
        f"{'Entropy':>8} | {'Phi abs':>8} | {'Corr':>8}"
    )
    sep = "-" * len(header)

    lines = [
        "",
        "=" * 70,
        "COMPARISON TABLE",
        "=" * 70,
        f"Warmup (epoch 50): Val {warmup_val:.2f}% | Test {warmup_test:.2f}%",
        "",
        header,
        sep,
    ]

    for name in CONFIGS:
        r = results.get(name)
        if r is None:
            lines.append(f"{name:<16} | {'FAILED':>10} |")
            continue
        ent = f"{r['final_entropy']:.3f}" if r["final_entropy"] is not None else "-"
        phi = f"{r['final_phi_abs_mean']:.4f}" if r["final_phi_abs_mean"] is not None else "-"
        corr = f"{r['final_corr']:+.3f}" if r["final_corr"] is not None else "-"
        lines.append(
            f"{name:<16} | {r['final_test']:>9.2f}% | {r['best_test']:>9.2f}% | "
            f"{ent:>8} | {phi:>8} | {corr:>8}"
        )

    lines.append(sep)
    text = "\n".join(lines)
    print(text)

    with open("results/exp2_summary.txt", "w") as f:
        f.write(text + "\n")


def plot_comparison(results):
    # Separate ALA experiments from baseline
    ala_names = [n for n in CONFIGS if CONFIGS[n].get("use_ala", False) and results.get(n)]
    all_names = [n for n in CONFIGS if results.get(n)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Exp2: ALA Ablation Comparison", fontsize=14)

    # 1. Test accuracy curves
    ax = axes[0, 0]
    for n in all_names:
        r = results[n]
        epochs = range(WARMUP_EPOCHS + 1, WARMUP_EPOCHS + 1 + len(r["test_accs"]))
        style = "--" if n == "baseline_ce" else "-"
        ax.plot(epochs, r["test_accs"], style, label=n, alpha=0.8)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlabel("Epoch")
    ax.set_title("Test Accuracy")
    ax.legend(fontsize=7, loc="lower right")

    # 2. Policy entropy
    ax = axes[0, 1]
    for n in ala_names:
        r = results[n]
        ax.plot(r["rl_log"]["entropy"], label=n, alpha=0.8)
    ax.axhline(y=1.099, color="r", linestyle="--", alpha=0.3, label="max (log3)")
    ax.set_ylabel("Entropy")
    ax.set_xlabel("K-window")
    ax.set_title("Policy Entropy")
    ax.legend(fontsize=7)

    # 3. Phi abs_mean
    ax = axes[1, 0]
    for n in ala_names:
        r = results[n]
        ax.plot(r["rl_log"]["phi_abs_mean"], label=n, alpha=0.8)
    ax.set_ylabel("Phi off-diag abs mean")
    ax.set_xlabel("K-window")
    ax.set_title("Phi Movement")
    ax.legend(fontsize=7)

    # 4. Confusion-Phi correlation
    ax = axes[1, 1]
    for n in ala_names:
        r = results[n]
        ax.plot(r["rl_log"]["corr"], label=n, alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("K-window")
    ax.set_title("Confusion–Phi Correlation")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("results/exp2_comparison.png", dpi=150)
    plt.close()
    print("Comparison plot saved: results/exp2_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("results", exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()

    print("Pre-loading val/test data onto GPU...")
    val_imgs = torch.cat([x for x, _ in val_loader]).to(device)
    val_tgts = torch.cat([y for _, y in val_loader]).to(device)
    test_imgs = torch.cat([x for x, _ in test_loader]).to(device)
    test_tgts = torch.cat([y for _, y in test_loader]).to(device)
    print(f"  val: {val_imgs.shape}, test: {test_imgs.shape}")

    # Phase 0: Warmup
    ckpt_path, warmup_val, warmup_test = run_warmup(
        train_loader, val_imgs, val_tgts, test_imgs, test_tgts, device,
    )

    # Phase 1: Experiments
    results = {}
    exp_names = list(CONFIGS.keys())
    for idx, name in enumerate(exp_names):
        cfg = CONFIGS[name]
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(exp_names)}] {name}: {cfg['desc']}")
        print(f"{'='*70}")

        r = run_experiment(
            name, cfg, ckpt_path,
            train_loader, val_imgs, val_tgts, test_imgs, test_tgts,
            device,
        )
        results[name] = r

        if r is not None:
            print(f"  → Final: {r['final_test']:.2f}% | Best: {r['best_test']:.2f}%")
        else:
            print(f"  → FAILED")

    # Phase 2: Comparison
    print_comparison(results, warmup_val, warmup_test)
    plot_comparison(results)

    print("\nDone! Check results/ directory.")


if __name__ == "__main__":
    main()
