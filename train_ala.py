"""Part 4c: Full ALA training loop following Algorithm 1 of the paper.

Training is structured around K-step windows (K=200). The first 50 epochs
use standard CE loss (warmup). After warmup, every SGD step within a K-window
is followed by a fast GPU-resident validation evaluation to compute the
cumulative discounted metric (Eq.4). At window boundaries the RL controller
computes reward (Eq.5), updates the policy, and adjusts Phi.

GPU memory optimization: val/test sets are pre-loaded onto GPU to eliminate
CPU-GPU transfer overhead during the ~200 val evals per K-window.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from adaptive_loss import AdaptiveLoss
from state import construct_states, get_pair_indices_tensor
from controller import (
    ALAPolicy,
    ReplayMemory,
    compute_reward,
)


# ---------------------------------------------------------------------------
# Fast GPU-resident evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def fast_evaluate(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 2048,
) -> float:
    """Compute accuracy using GPU-resident data. No CPU-GPU transfer."""
    model.eval()
    correct = 0
    with torch.cuda.amp.autocast():
        for i in range(0, len(images), batch_size):
            logits = model(images[i:i + batch_size])
            correct += (logits.argmax(1) == targets[i:i + batch_size]).sum().item()
    return correct / len(images) * 100.0


@torch.no_grad()
def fast_confusion_matrix(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Compute Eq.7 confusion matrix using GPU-resident data."""
    model.eval()
    C = torch.zeros(num_classes, num_classes, device=images.device)
    class_counts = torch.zeros(num_classes, device=images.device)

    with torch.cuda.amp.autocast():
        for i in range(0, len(images), batch_size):
            logits = model(images[i:i + batch_size])
            neg_log_probs = -F.log_softmax(logits.float(), dim=1)
            batch_targets = targets[i:i + batch_size]

            C.index_add_(0, batch_targets, neg_log_probs)
            class_counts.scatter_add_(
                0, batch_targets,
                torch.ones(batch_targets.size(0), device=images.device),
            )

    nonzero = class_counts > 0
    C[nonzero] /= class_counts[nonzero].unsqueeze(1)
    return C


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def actions_to_delta_phi(
    actions: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    num_classes: int,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Convert action indices to a symmetric delta_phi matrix."""
    action_map = torch.tensor([-beta, 0.0, beta], device=device)
    delta_values = action_map[actions]

    delta_phi = torch.zeros(num_classes, num_classes, device=device)
    delta_phi[pair_i, pair_j] = delta_values
    delta_phi[pair_j, pair_i] = delta_values
    return delta_phi


def update_policy(
    policy: ALAPolicy,
    optimizer: torch.optim.Optimizer,
    memory: ReplayMemory,
    batch_size: int,
    baseline_ema: float,
    device: torch.device,
) -> float:
    """REINFORCE policy gradient update with EMA baseline."""
    samples = memory.sample(batch_size)

    total_loss = torch.tensor(0.0, device=device)
    total_reward = 0.0

    for states, actions, _old_log_probs, reward in samples:
        states = states.to(device)
        actions = actions.to(device)

        logits = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)

        advantage = reward - baseline_ema
        sample_loss = -(new_log_probs * advantage).mean()
        total_loss = total_loss + sample_loss
        total_reward += reward

    total_loss = total_loss / len(samples)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    avg_reward = total_reward / len(samples)
    baseline_ema = 0.9 * baseline_ema + 0.1 * avg_reward
    return baseline_ema


def log_and_print(msg: str, log_file) -> None:
    tqdm.write(msg)
    log_file.write(msg + "\n")
    log_file.flush()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data, model, optimizer
    train_loader, val_loader, test_loader = get_dataloaders()
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)

    # Loss functions
    num_classes = 100
    ce_criterion = nn.CrossEntropyLoss()
    adaptive_loss = AdaptiveLoss(num_classes).to(device)
    warmup_epochs = 50

    # RL controller
    state_dim = 24
    policy = ALAPolicy(state_dim).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    memory = ReplayMemory(1000)

    # Hyperparameters
    K = 200
    beta = 0.1
    gamma = 0.9
    policy_batch_size = 8
    pair_i, pair_j = get_pair_indices_tensor(num_classes, device)

    steps_per_epoch = len(train_loader)
    total_steps = 200 * steps_per_epoch

    # -----------------------------------------------------------------------
    # GPU preload val/test sets (~240 MB total on A100 40GB)
    # -----------------------------------------------------------------------
    print("Pre-loading val/test data onto GPU...")
    val_images = torch.cat([x for x, _ in val_loader]).to(device)
    val_targets = torch.cat([y for _, y in val_loader]).to(device)
    test_images = torch.cat([x for x, _ in test_loader]).to(device)
    test_targets = torch.cat([y for _, y in test_loader]).to(device)
    print(f"  val: {val_images.shape}, test: {test_images.shape}")

    # State tracking
    confusion_history: list[torch.Tensor] = []
    M_old: float | None = None
    prev_states: torch.Tensor | None = None
    prev_actions: torch.Tensor | None = None
    prev_log_probs: torch.Tensor | None = None
    baseline_ema = 0.0

    # Logging
    os.makedirs("results/curves", exist_ok=True)
    os.makedirs("results/phi_heatmaps", exist_ok=True)

    train_losses: list[float] = []
    val_accs: list[float] = []
    test_accs: list[float] = []

    log_file = open("results/ala_training_log.txt", "w")

    # K-window based training loop
    global_step = 0
    current_epoch = 0
    epoch_loss = 0.0
    epoch_total = 0
    last_grad_norm = 0.0
    train_iter = iter(train_loader)

    pbar = tqdm(total=total_steps, desc="ALA Training", unit="step")

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

                if current_epoch <= warmup_epochs:
                    log_and_print(
                        f"Epoch {current_epoch:3d} | CE Warmup | "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%",
                        log_file,
                    )
                else:
                    log_and_print(
                        f"Epoch {current_epoch:3d} | Train Loss: {avg_loss:.4f} | "
                        f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%",
                        log_file,
                    )

                if current_epoch == warmup_epochs:
                    log_and_print(
                        "CE warmup complete. Switching to adaptive loss + RL.",
                        log_file,
                    )

                # Debug: gradient norm, inner product, phi stats
                model.eval()
                with torch.no_grad():
                    dbg_logits = model(val_images[:2048])
                    dbg_log_probs = F.log_softmax(dbg_logits, dim=1)
                    dbg_y = F.one_hot(val_targets[:2048], num_classes).float()
                    dbg_weighted = dbg_y @ adaptive_loss.phi.data
                    dbg_inner = (dbg_weighted * dbg_log_probs).sum(dim=1)
                    dbg_sig_mean = torch.sigmoid(dbg_inner).mean()
                dbg_mask = ~torch.eye(num_classes, dtype=torch.bool, device=device)
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

                if current_epoch in (50, 100, 150, 200):
                    torch.save(
                        adaptive_loss.phi.data.cpu(),
                        f"results/phi_heatmaps/phi_epoch_{current_epoch:03d}.pt",
                    )

                scheduler.step()
                epoch_loss = 0.0
                epoch_total = 0
                train_iter = iter(train_loader)
                inputs, targets_batch = next(train_iter)

            inputs, targets_batch = inputs.to(device), targets_batch.to(device)

            # Determine current epoch for loss selection
            approx_epoch = global_step // steps_per_epoch + 1

            # SGD step
            model.train()
            optimizer.zero_grad()
            logits = model(inputs)

            if approx_epoch <= warmup_epochs:
                loss = ce_criterion(logits, targets_batch)
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

            # Eq.4: measure val error at every step (only after warmup)
            if approx_epoch > warmup_epochs:
                val_acc_step = fast_evaluate(model, val_images, val_targets)
                val_error_step = 100.0 - val_acc_step
                weight = gamma ** (K - j)
                cumulative_metric += weight * val_error_step
                model.train()

            if global_step >= total_steps:
                break

        # === End of K-step window ===
        approx_epoch = global_step // steps_per_epoch + 1

        if approx_epoch > warmup_epochs:
            # RL active: compute reward, update policy, adjust Phi
            M_new = cumulative_metric

            # Reward (Eq.5) — from second window onward
            if M_old is not None and prev_states is not None:
                reward = compute_reward(M_old, M_new)
                memory.push(prev_states, prev_actions, prev_log_probs, reward)
                reward_val_error = 100.0 - fast_evaluate(model, val_images, val_targets)
                _reward_str = (
                    f"  [Step {global_step}] Reward: {reward:+.1f} | "
                    f"Val Error: {reward_val_error:.2f}%"
                )
                # will be extended with action/row stats below

            # Policy update
            if len(memory) >= policy_batch_size:
                baseline_ema = update_policy(
                    policy, policy_optimizer, memory,
                    policy_batch_size, baseline_ema, device,
                )

            # Confusion matrix → state → action → Φ update
            progress = global_step / total_steps
            C = fast_confusion_matrix(model, val_images, val_targets, num_classes)
            confusion_history.append(C)
            if len(confusion_history) > 10:
                confusion_history.pop(0)

            all_states = construct_states(
                confusion_history, adaptive_loss.phi.data, progress,
                num_classes, pair_i, pair_j,
            )

            with torch.no_grad():
                actions, log_probs = policy.select_action(all_states)

            delta_phi = actions_to_delta_phi(
                actions, pair_i, pair_j, num_classes, beta, device,
            )

            # Action distribution
            n_plus = (actions == 2).sum().item()
            n_zero = (actions == 1).sum().item()
            n_minus = (actions == 0).sum().item()
            total_actions = actions.numel()
            # Phi row-mean stats
            phi_row_tmp = adaptive_loss.phi.data.clone()
            phi_row_tmp.fill_diagonal_(0)
            row_means = phi_row_tmp.sum(dim=1) / (num_classes - 1)
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

            adaptive_loss.update_phi(delta_phi * 0.001)

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

        if current_epoch in (50, 100, 150, 200):
            torch.save(
                adaptive_loss.phi.data.cpu(),
                f"results/phi_heatmaps/phi_epoch_{current_epoch:03d}.pt",
            )

    log_file.close()

    # Save curves
    num_epochs = len(train_losses)
    epochs = range(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("ALA Train Loss")
    plt.savefig("results/curves/ala_train_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, test_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("ALA Test Accuracy")
    plt.savefig("results/curves/ala_test_acc.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy (%)")
    plt.title("ALA Validation Accuracy")
    plt.savefig("results/curves/ala_val_acc.png", dpi=150)
    plt.close()

    print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%")


if __name__ == "__main__":
    main()
