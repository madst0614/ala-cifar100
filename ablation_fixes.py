"""Ablation study: 3 experiments with shared CE warmup checkpoint.

Experiments:
  1. CE blend:              loss = 0.5*CE + 0.5*adaptive
  2. Reward Phi penalty:    reward = sign(M_old-M_new) - 0.1*mean(|phi_offdiag|)
  3. CE blend + Phi penalty: both applied

Common: delta_phi*0.01, clamp [-0.1, 0.1], no decay, 200 epochs total.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

# ── Constants ──────────────────────────────────────────────────────────────
WARMUP_EPOCHS = 50
TOTAL_EPOCHS = 200
NUM_CLASSES = 100
K = 200
BETA = 0.1
GAMMA = 0.9
POLICY_BATCH_SIZE = 8
PHI_SCALE = 0.01
CKPT_PATH = "results/ablation_warmup.pt"


# ── Warmup ─────────────────────────────────────────────────────────────────

def run_warmup(train_loader, val_images, val_targets, test_images, test_targets,
               device, log_file):
    """Train 50 epochs with CE loss, save model/optimizer/scheduler checkpoint."""
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    ce_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, WARMUP_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_total = 0
        last_grad_norm = 0.0

        for inputs, targets_batch in train_loader:
            inputs, targets_batch = inputs.to(device), targets_batch.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = ce_criterion(logits, targets_batch)
            loss.backward()
            last_grad_norm = (sum(
                p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
            ) ** 0.5).item()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            epoch_total += inputs.size(0)

        scheduler.step()
        avg_loss = epoch_loss / epoch_total
        val_acc = fast_evaluate(model, val_images, val_targets)
        test_acc = fast_evaluate(model, test_images, test_targets)

        log_and_print(
            f"Epoch {epoch:3d} | CE Warmup | Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%",
            log_file,
        )

        # Debug logging (phi is identity during warmup)
        phi_id = torch.eye(NUM_CLASSES, device=device)
        model.eval()
        with torch.no_grad():
            dbg_logits = model(val_images[:2048])
            dbg_log_probs = F.log_softmax(dbg_logits, dim=1)
            dbg_y = F.one_hot(val_targets[:2048], NUM_CLASSES).float()
            dbg_inner = (dbg_y @ phi_id * dbg_log_probs).sum(dim=1)
            dbg_sig_mean = torch.sigmoid(dbg_inner).mean()
        dbg_mask = ~torch.eye(NUM_CLASSES, dtype=torch.bool, device=device)
        dbg_off = phi_id[dbg_mask]
        log_and_print(
            f"  [Debug] GradNorm: {last_grad_norm:.4f} | "
            f"Inner: mean={dbg_inner.mean():.2f} std={dbg_inner.std():.2f} "
            f"min={dbg_inner.min():.2f} max={dbg_inner.max():.2f} | "
            f"Sig: {dbg_sig_mean:.4f} | "
            f"Phi: mean={dbg_off.mean():.4f} std={dbg_off.std():.4f} "
            f"[{dbg_off.min():.4f}, {dbg_off.max():.4f}]",
            log_file,
        )

    log_and_print("CE warmup complete. Saving checkpoint.", log_file)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, CKPT_PATH)


# ── Single experiment ──────────────────────────────────────────────────────

def run_experiment(name, ce_blend, phi_penalty,
                   train_loader, val_images, val_targets,
                   test_images, test_targets, device, log_file):
    """Run one ablation experiment from the warmup checkpoint.

    Returns final test accuracy.
    """
    set_seed(42)

    # Rebuild model/optimizer/scheduler and load checkpoint
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    # Fresh RL components
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
    M_old: float | None = None
    prev_states: torch.Tensor | None = None
    prev_actions: torch.Tensor | None = None
    prev_log_probs: torch.Tensor | None = None
    baseline_ema = 0.0

    train_losses: list[float] = []
    val_accs: list[float] = []
    test_accs: list[float] = []

    global_step = warmup_steps
    current_epoch = WARMUP_EPOCHS
    epoch_loss = 0.0
    epoch_total = 0
    last_grad_norm = 0.0
    train_iter = iter(train_loader)

    pbar = tqdm(
        total=total_steps - warmup_steps, desc=name, unit="step",
    )

    while global_step < total_steps:
        # === One K-step window ===
        cumulative_metric = 0.0

        for j in range(1, K + 1):
            # Get next training batch (handle epoch boundary)
            try:
                inputs, targets_batch = next(train_iter)
            except StopIteration:
                # Epoch complete — log, scheduler step, reset
                current_epoch += 1
                avg_loss = epoch_loss / max(epoch_total, 1)
                val_acc = fast_evaluate(model, val_images, val_targets)
                test_acc = fast_evaluate(model, test_images, test_targets)

                train_losses.append(avg_loss)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                log_and_print(
                    f"Epoch {current_epoch:3d} | Train Loss: {avg_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%",
                    log_file,
                )

                # Debug: gradient norm, inner product, phi stats
                model.eval()
                with torch.no_grad():
                    dbg_logits = model(val_images[:2048])
                    dbg_log_probs = F.log_softmax(dbg_logits, dim=1)
                    dbg_y = F.one_hot(val_targets[:2048], NUM_CLASSES).float()
                    dbg_weighted = dbg_y @ adaptive_loss.phi.data
                    dbg_inner = (dbg_weighted * dbg_log_probs).sum(dim=1)
                    dbg_sig_mean = torch.sigmoid(dbg_inner).mean()
                dbg_mask = ~torch.eye(NUM_CLASSES, dtype=torch.bool, device=device)
                dbg_off = adaptive_loss.phi.data[dbg_mask]
                log_and_print(
                    f"  [Debug] GradNorm: {last_grad_norm:.4f} | "
                    f"Inner: mean={dbg_inner.mean():.2f} std={dbg_inner.std():.2f} "
                    f"min={dbg_inner.min():.2f} max={dbg_inner.max():.2f} | "
                    f"Sig: {dbg_sig_mean:.4f} | "
                    f"Phi: mean={dbg_off.mean():.4f} std={dbg_off.std():.4f} "
                    f"[{dbg_off.min():.4f}, {dbg_off.max():.4f}]",
                    log_file,
                )

                scheduler.step()
                epoch_loss = 0.0
                epoch_total = 0
                train_iter = iter(train_loader)
                inputs, targets_batch = next(train_iter)

            inputs, targets_batch = inputs.to(device), targets_batch.to(device)

            # SGD step
            model.train()
            optimizer.zero_grad()
            logits = model(inputs)

            if ce_blend:
                loss = (0.5 * F.cross_entropy(logits, targets_batch)
                        + 0.5 * adaptive_loss(logits, targets_batch))
            else:
                loss = adaptive_loss(logits, targets_batch)

            loss.backward()
            last_grad_norm = (sum(
                p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
            ) ** 0.5).item()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_total += inputs.size(0)
            global_step += 1
            pbar.update(1)

            # Eq.4: measure val error at every step
            val_acc_step = fast_evaluate(model, val_images, val_targets)
            val_error_step = 100.0 - val_acc_step
            weight = GAMMA ** (K - j)
            cumulative_metric += weight * val_error_step
            model.train()

            if global_step >= total_steps:
                break

        # === End of K-step window: RL update ===
        M_new = cumulative_metric

        # Reward (Eq.5) — from second window onward
        if M_old is not None and prev_states is not None:
            reward = compute_reward(M_old, M_new)

            if phi_penalty:
                mask = ~torch.eye(NUM_CLASSES, dtype=torch.bool, device=device)
                phi_pen = adaptive_loss.phi.data[mask].abs().mean().item()
                reward = reward - 0.1 * phi_pen

            memory.push(prev_states, prev_actions, prev_log_probs, reward)
            reward_val_error = 100.0 - fast_evaluate(model, val_images, val_targets)
            _reward_str = (
                f"  [Step {global_step}] Reward: {reward:+.4f} | "
                f"Val Error: {reward_val_error:.2f}%"
            )

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

        with torch.no_grad():
            actions, log_probs = policy.select_action(all_states)

        delta_phi = actions_to_delta_phi(
            actions, pair_i, pair_j, NUM_CLASSES, BETA, device,
        )

        # Action distribution
        n_plus = (actions == 2).sum().item()
        n_zero = (actions == 1).sum().item()
        n_minus = (actions == 0).sum().item()
        total_actions = actions.numel()
        # Phi row-mean stats
        phi_row_tmp = adaptive_loss.phi.data.clone()
        phi_row_tmp.fill_diagonal_(0)
        row_means = phi_row_tmp.sum(dim=1) / (NUM_CLASSES - 1)
        row_mean_min = row_means.min().item()
        row_mean_max = row_means.max().item()
        row_mean_std = row_means.std().item()
        if M_old is not None and prev_states is not None:
            log_and_print(
                _reward_str
                + f" | Actions: +={n_plus}({n_plus/total_actions*100:.0f}%)"
                f" 0={n_zero}({n_zero/total_actions*100:.0f}%)"
                f" -={n_minus}({n_minus/total_actions*100:.0f}%)"
                f" | RowMean: [{row_mean_min:.4f}, {row_mean_max:.4f}]"
                f" std={row_mean_std:.4f}",
                log_file,
            )

        adaptive_loss.update_phi(delta_phi * PHI_SCALE)

        M_old = M_new
        prev_states = all_states
        prev_actions = actions
        prev_log_probs = log_probs

    pbar.close()

    # Log final epoch if there are pending samples
    if epoch_total > 0:
        current_epoch += 1
        avg_loss = epoch_loss / epoch_total
        val_acc = fast_evaluate(model, val_images, val_targets)
        test_acc = fast_evaluate(model, test_images, test_targets)

        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        log_and_print(
            f"Epoch {current_epoch:3d} | Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%",
            log_file,
        )

    final_acc = test_accs[-1] if test_accs else fast_evaluate(
        model, test_images, test_targets,
    )
    return final_acc


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()

    print("Pre-loading val/test data onto GPU...")
    val_images = torch.cat([x for x, _ in val_loader]).to(device)
    val_targets = torch.cat([y for _, y in val_loader]).to(device)
    test_images = torch.cat([x for x, _ in test_loader]).to(device)
    test_targets = torch.cat([y for _, y in test_loader]).to(device)
    print(f"  val: {val_images.shape}, test: {test_images.shape}")

    os.makedirs("results", exist_ok=True)
    log_file = open("results/ablation_log.txt", "w")

    # ── Phase 0: Shared CE warmup ──────────────────────────────────────
    log_and_print("=" * 60, log_file)
    log_and_print("Phase 0: CE Warmup (50 epochs)", log_file)
    log_and_print("=" * 60, log_file)
    run_warmup(
        train_loader, val_images, val_targets, test_images, test_targets,
        device, log_file,
    )

    # ── Phase 1: Ablation experiments ──────────────────────────────────
    experiments = [
        ("Exp1: CE blend",                      True,  False),
        ("Exp2: Reward Phi penalty",             False, True),
        ("Exp3: CE blend + Reward Phi penalty",  True,  True),
    ]

    results = {}
    for name, ce_blend, phi_pen in experiments:
        log_and_print("=" * 60, log_file)
        log_and_print(name, log_file)
        log_and_print("=" * 60, log_file)

        acc = run_experiment(
            name, ce_blend, phi_pen,
            train_loader, val_images, val_targets,
            test_images, test_targets, device, log_file,
        )
        results[name] = acc
        log_and_print(f"\n{name} — final test accuracy: {acc:.2f}%\n", log_file)

    # ── Summary table ──────────────────────────────────────────────────
    log_and_print("=" * 60, log_file)
    log_and_print("ABLATION SUMMARY", log_file)
    log_and_print("=" * 60, log_file)
    log_and_print(f"{'Experiment':<42} {'Test Acc':>10}", log_file)
    log_and_print("-" * 54, log_file)
    for name, acc in results.items():
        log_and_print(f"{name:<42} {acc:>9.2f}%", log_file)
    log_and_print("=" * 60, log_file)

    log_file.close()


if __name__ == "__main__":
    main()
