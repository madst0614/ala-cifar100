"""Part 4c: Full ALA training loop with RL-controlled adaptive loss."""

import os

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    set_seed,
    get_dataloaders,
    get_model,
    evaluate,
    get_optimizer_and_scheduler,
)
from adaptive_loss import AdaptiveLoss
from state import compute_confusion_matrix, construct_states, get_pair_indices_tensor
from controller import ALAPolicy, ReplayMemory, compute_reward


def actions_to_delta_phi(
    actions: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    num_classes: int,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Convert action indices to a delta_phi matrix.

    Args:
        actions: (num_pairs,) indices in {0, 1, 2} mapping to {-beta, 0, +beta}.
        pair_i: (num_pairs,) tensor of row indices.
        pair_j: (num_pairs,) tensor of column indices.
        num_classes: number of classes.
        beta: step size for phi updates.
        device: torch device.

    Returns:
        (num_classes, num_classes) symmetric delta_phi tensor.
    """
    action_map = torch.tensor([-beta, 0.0, beta], device=device)
    delta_values = action_map[actions]  # (num_pairs,)

    delta_phi = torch.zeros(num_classes, num_classes, device=device)
    delta_phi[pair_i, pair_j] = delta_values
    delta_phi[pair_j, pair_i] = delta_values  # symmetry

    return delta_phi


def update_policy(
    policy: ALAPolicy,
    optimizer: torch.optim.Optimizer,
    memory: ReplayMemory,
    batch_size: int,
    baseline_ema: float,
    device: torch.device,
) -> float:
    """REINFORCE policy gradient update.

    Re-computes log_probs via a fresh forward pass so that gradients
    flow back through the policy network.

    Args:
        policy: the policy network.
        optimizer: policy optimizer.
        memory: replay memory.
        batch_size: number of transitions to sample.
        baseline_ema: exponential moving average baseline for variance reduction.
        device: torch device.

    Returns:
        Updated baseline_ema.
    """
    samples = memory.sample(batch_size)

    total_loss = torch.tensor(0.0, device=device)
    total_reward = 0.0

    for states, actions, _old_log_probs, reward in samples:
        states = states.to(device)
        actions = actions.to(device)

        # Re-forward through policy to get fresh log_probs with grad
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

    # Update baseline EMA
    avg_reward = total_reward / len(samples)
    baseline_ema = 0.9 * baseline_ema + 0.1 * avg_reward

    return baseline_ema


def log_and_print(msg: str, log_file) -> None:
    """Print message and write to log file."""
    tqdm.write(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def main() -> None:
    """Run full ALA training: ResNet-18 + adaptive loss + RL controller."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data, model, optimizer
    train_loader, val_loader, test_loader = get_dataloaders()
    model = get_model(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)

    # Adaptive loss
    num_classes = 100
    adaptive_loss = AdaptiveLoss(num_classes).to(device)

    # RL controller
    state_dim = 24
    policy = ALAPolicy(state_dim).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    memory = ReplayMemory(1000)

    # Hyperparameters
    K = 200
    beta = 0.1
    policy_batch_size = 8
    warmup_epochs = 10  # First 10 epochs: Φ=I fixed, no RL
    pair_i, pair_j = get_pair_indices_tensor(num_classes, device)

    total_iterations = 200 * len(train_loader)

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
    global_step = 0

    for epoch in tqdm(range(1, 201), desc="ALA Training"):
        model.train()
        running_loss = 0.0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = adaptive_loss(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            global_step += 1

            # RL controller update every K steps (skip during warmup)
            if global_step % K == 0 and epoch > warmup_epochs:
                progress = global_step / total_iterations

                # 1. Validation error
                val_acc = evaluate(model, val_loader, device)
                val_error = 100.0 - val_acc
                M_new = val_error

                # 2. Reward computation (from second update onward)
                if M_old is not None and prev_states is not None:
                    reward = compute_reward(M_old, M_new)
                    memory.push(prev_states, prev_actions, prev_log_probs, reward)
                    log_and_print(
                        f"  [Step {global_step}] Reward: {reward:+.1f} | "
                        f"Val Error: {val_error:.2f}%",
                        log_file,
                    )

                # 3. Policy update
                if len(memory) >= policy_batch_size:
                    baseline_ema = update_policy(
                        policy, policy_optimizer, memory,
                        policy_batch_size, baseline_ema, device,
                    )

                # 4. Confusion matrix → state
                C = compute_confusion_matrix(model, val_loader, num_classes, device)
                confusion_history.append(C)
                if len(confusion_history) > 10:
                    confusion_history.pop(0)

                states = construct_states(
                    confusion_history, adaptive_loss.phi.data, progress,
                    num_classes, pair_i, pair_j,
                )

                # 5. Action sampling
                with torch.no_grad():
                    actions, action_log_probs = policy.select_action(states)

                # 6. Phi update
                delta_phi = actions_to_delta_phi(
                    actions, pair_i, pair_j, num_classes, beta, device,
                )
                adaptive_loss.update_phi(delta_phi)

                # 7. Save state for next reward computation
                M_old = M_new
                prev_states = states
                prev_actions = actions
                prev_log_probs = action_log_probs

                # Restore train mode
                model.train()

        # End of epoch
        epoch_loss = running_loss / total
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        train_losses.append(epoch_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        log_and_print(
            f"Epoch {epoch:3d} | Train Loss: {epoch_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%",
            log_file,
        )

        # Save phi at checkpoints
        if epoch in (50, 100, 150, 200):
            torch.save(
                adaptive_loss.phi.data.cpu(),
                f"results/phi_heatmaps/phi_epoch_{epoch:03d}.pt",
            )

    log_file.close()

    # Save curves
    epochs = range(1, 201)

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
