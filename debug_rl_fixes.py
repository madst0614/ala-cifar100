"""Debug RL fixes: 4 experiments testing different RL stabilization strategies.

Each experiment: CE warmup 10 epochs → adaptive loss + RL 20 epochs (30 total).
GPU-preloaded val/test data is shared across all experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from utils import set_seed, get_dataloaders, get_model, get_optimizer_and_scheduler
from adaptive_loss import AdaptiveLoss
from state import construct_states, get_pair_indices_tensor
from controller import ALAPolicy, ReplayMemory, compute_reward


# ---------------------------------------------------------------------------
# Helpers (from train_ala.py)
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
            batch_targets = targets[i:i + batch_size]
            C.index_add_(0, batch_targets, neg_log_probs)
            class_counts.scatter_add_(
                0, batch_targets,
                torch.ones(batch_targets.size(0), device=images.device),
            )
    nonzero = class_counts > 0
    C[nonzero] /= class_counts[nonzero].unsqueeze(1)
    return C


def actions_to_delta_phi(actions, pair_i, pair_j, num_classes, beta, device):
    action_map = torch.tensor([-beta, 0.0, beta], device=device)
    delta_values = action_map[actions]
    delta_phi = torch.zeros(num_classes, num_classes, device=device)
    delta_phi[pair_i, pair_j] = delta_values
    delta_phi[pair_j, pair_i] = delta_values
    return delta_phi


def update_policy(policy, optimizer, memory, batch_size, baseline_ema, device):
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


# ---------------------------------------------------------------------------
# Temperature policy for Fix 1
# ---------------------------------------------------------------------------

class TemperaturePolicy(ALAPolicy):
    def select_action(self, state, temperature=5.0):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.forward(state) / temperature
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    fix_name: str,
    device: torch.device,
    train_loader,
    val_images, val_targets,
    test_images, test_targets,
    make_policy_fn,
    apply_phi_fn,
):
    """Run one ALA experiment with a specific fix applied.

    Args:
        make_policy_fn: callable(state_dim, device) -> policy
        apply_phi_fn: callable(adaptive_loss, delta_phi, device, num_classes) -> None
    """
    print("=" * 60)
    print(f"Fix: {fix_name}")
    print("=" * 60)

    set_seed(42)
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, T_max=30)

    num_classes = 100
    warmup_epochs = 10
    ce_criterion = nn.CrossEntropyLoss()
    adaptive_loss = AdaptiveLoss(num_classes).to(device)

    state_dim = 24
    policy = make_policy_fn(state_dim, device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    memory = ReplayMemory(1000)

    K = 200
    beta = 0.1
    gamma = 0.9
    policy_batch_size = 8
    pair_i, pair_j = get_pair_indices_tensor(num_classes, device)

    steps_per_epoch = len(train_loader)
    total_steps = 30 * steps_per_epoch

    confusion_history: list[torch.Tensor] = []
    M_old: float | None = None
    prev_states: torch.Tensor | None = None
    prev_actions: torch.Tensor | None = None
    prev_log_probs: torch.Tensor | None = None
    baseline_ema = 0.0

    reward_counts = {+1.0: 0, 0.0: 0, -1.0: 0}

    global_step = 0
    current_epoch = 0
    epoch_loss = 0.0
    epoch_total = 0
    train_iter = iter(train_loader)

    pbar = tqdm(total=total_steps, desc=fix_name, unit="step")

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

                phase = "CE" if current_epoch <= warmup_epochs else "ADA+RL"
                tqdm.write(
                    f"  Epoch {current_epoch:3d} [{phase:6s}] | "
                    f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
                )

                scheduler.step()
                epoch_loss = 0.0
                epoch_total = 0
                train_iter = iter(train_loader)
                inputs, targets_batch = next(train_iter)

            inputs, targets_batch = inputs.to(device), targets_batch.to(device)
            approx_epoch = global_step // steps_per_epoch + 1

            model.train()
            optimizer.zero_grad()
            logits = model(inputs)

            if approx_epoch <= warmup_epochs:
                loss = ce_criterion(logits, targets_batch)
            else:
                loss = adaptive_loss(logits, targets_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_total += inputs.size(0)
            global_step += 1
            pbar.update(1)

            if approx_epoch > warmup_epochs:
                val_acc_step = fast_evaluate(model, val_images, val_targets)
                val_error_step = 100.0 - val_acc_step
                weight = gamma ** (K - j)
                cumulative_metric += weight * val_error_step
                model.train()

            if global_step >= total_steps:
                break

        # End of K-window
        approx_epoch = global_step // steps_per_epoch + 1

        if approx_epoch > warmup_epochs:
            M_new = cumulative_metric

            if M_old is not None and prev_states is not None:
                reward = compute_reward(M_old, M_new)
                memory.push(prev_states, prev_actions, prev_log_probs, reward)
                reward_counts[reward] = reward_counts.get(reward, 0) + 1

            if len(memory) >= policy_batch_size:
                baseline_ema = update_policy(
                    policy, policy_optimizer, memory,
                    policy_batch_size, baseline_ema, device,
                )

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

            # Apply the fix-specific phi update
            apply_phi_fn(adaptive_loss, delta_phi, device, num_classes)

            M_old = M_new
            prev_states = all_states
            prev_actions = actions
            prev_log_probs = log_probs

    pbar.close()

    # Final epoch stats
    if epoch_total > 0:
        current_epoch += 1
        avg_loss = epoch_loss / epoch_total
        val_acc = fast_evaluate(model, val_images, val_targets)
        test_acc = fast_evaluate(model, test_images, test_targets)
        tqdm.write(
            f"  Epoch {current_epoch:3d} [ADA+RL] | "
            f"Loss: {avg_loss:.4f} | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%"
        )

    val_acc = fast_evaluate(model, val_images, val_targets)
    test_acc = fast_evaluate(model, test_images, test_targets)
    total_rewards = sum(reward_counts.values())
    print(f"  Final: Val={val_acc:.2f}%, Test={test_acc:.2f}%")
    if total_rewards > 0:
        print(
            f"  Rewards: +1={reward_counts.get(1.0,0)}/{total_rewards} "
            f"({reward_counts.get(1.0,0)/total_rewards*100:.0f}%), "
            f"0={reward_counts.get(0.0,0)}/{total_rewards}, "
            f"-1={reward_counts.get(-1.0,0)}/{total_rewards} "
            f"({reward_counts.get(-1.0,0)/total_rewards*100:.0f}%)"
        )
    phi_offdiag = adaptive_loss.phi.data[
        ~torch.eye(num_classes, dtype=torch.bool, device=device)
    ]
    print(
        f"  Phi off-diag: mean={phi_offdiag.mean():.4f}, "
        f"std={phi_offdiag.std():.4f}, "
        f"min={phi_offdiag.min():.4f}, max={phi_offdiag.max():.4f}"
    )
    print()


# ---------------------------------------------------------------------------
# Fix-specific phi update functions
# ---------------------------------------------------------------------------

def apply_baseline(adaptive_loss, delta_phi, device, num_classes):
    """Fix 1 (temperature): standard phi update, policy handles the fix."""
    adaptive_loss.update_phi(delta_phi)


def apply_scaled_delta(adaptive_loss, delta_phi, device, num_classes):
    """Fix 2: scale delta_phi by 0.01."""
    adaptive_loss.update_phi(delta_phi * 0.01)


def apply_phi_decay(adaptive_loss, delta_phi, device, num_classes):
    """Fix 3: standard update + decay toward identity."""
    adaptive_loss.update_phi(delta_phi)
    diag = torch.eye(num_classes, device=device)
    adaptive_loss.phi.data = 0.95 * adaptive_loss.phi.data + 0.05 * diag
    adaptive_loss.phi.data.fill_diagonal_(1.0)


def apply_scaled_and_decay(adaptive_loss, delta_phi, device, num_classes):
    """Fix 4: scaled delta (0.01) + decay toward identity."""
    adaptive_loss.update_phi(delta_phi * 0.01)
    diag = torch.eye(num_classes, device=device)
    adaptive_loss.phi.data = 0.95 * adaptive_loss.phi.data + 0.05 * diag
    adaptive_loss.phi.data.fill_diagonal_(1.0)


# ---------------------------------------------------------------------------
# Policy factory functions
# ---------------------------------------------------------------------------

def make_standard_policy(state_dim, device):
    return ALAPolicy(state_dim).to(device)


def make_temperature_policy(state_dim, device):
    return TemperaturePolicy(state_dim).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # GPU preload (shared across all experiments)
    train_loader, val_loader, test_loader = get_dataloaders()
    print("Pre-loading val/test data onto GPU...")
    val_images = torch.cat([x for x, _ in val_loader]).to(device)
    val_targets = torch.cat([y for _, y in val_loader]).to(device)
    test_images = torch.cat([x for x, _ in test_loader]).to(device)
    test_targets = torch.cat([y for _, y in test_loader]).to(device)
    print(f"  val: {val_images.shape}, test: {test_images.shape}\n")

    experiments = [
        ("Fix 1: Temperature scaling (T=5.0)", make_temperature_policy, apply_baseline),
        ("Fix 2: Delta phi scaling (0.01x)", make_standard_policy, apply_scaled_delta),
        ("Fix 3: Phi decay toward identity (0.95)", make_standard_policy, apply_phi_decay),
        ("Fix 4: Scaled delta (0.01x) + Phi decay (0.95)", make_standard_policy, apply_scaled_and_decay),
    ]

    for fix_name, policy_fn, phi_fn in experiments:
        run_experiment(
            fix_name, device, train_loader,
            val_images, val_targets,
            test_images, test_targets,
            policy_fn, phi_fn,
        )

    print("=" * 60)
    print("All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
