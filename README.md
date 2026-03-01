# ALA (Adaptive Loss Alignment) — CIFAR-100

ResNet-18 on CIFAR-100 with RL-controlled adaptive loss function.

## Project Structure

```
├── utils.py           # Common utilities (data, model, eval, optimizer)
├── baseline.py        # Part 1: Standard cross-entropy training
├── adaptive_loss.py   # Part 2: Parametric loss function (Eq.6)
├── state.py           # Part 3: Confusion matrix & state representation
├── controller.py      # Part 4a/4b: Policy network, replay memory, reward
├── train_ala.py       # Part 4c: Full ALA training loop
├── analysis.py        # Part 5a: Phi heatmap visualization
├── answers.md         # Written answers for Parts 2a, 2c, 2d
└── results/
    ├── curves/        # Training loss and accuracy plots
    ├── phi_heatmaps/  # Phi matrices (.pt) and evolution plot
    └── ala_training_log.txt
```

## How to Run

```bash
# 1. Baseline training (standard cross-entropy, 200 epochs)
python baseline.py

# 2. ALA training (adaptive loss + RL controller, 200 epochs)
python train_ala.py

# 3. Phi visualization (after ALA training completes)
python analysis.py
```

## Key Design Decisions

- **State dimension**: 24 per class pair (10 timesteps x 2 confusion values + 2 relative change + 1 phi + 1 progress)
- **Reward**: `sign(M_old - M_new)` where M is validation classification error. Simplified to use the single val error at each controller step rather than discounted metrics over sub-intervals.
- **Replay memory**: Stored log-probs are used directly (slightly off-policy as the policy changes, but memory is small enough to stay approximately on-policy).
- **Phi constraints**: Symmetry enforced via `(Phi + Phi^T) / 2` after each update. Values clamped to `[-1, 1]`.
- **Controller frequency**: Every K=200 SGD steps, the controller evaluates on the validation set, computes reward, updates the policy, and adjusts Phi.

## File-to-Part Mapping

| File | Part |
|------|------|
| `baseline.py` | Part 1 — Baseline CE training |
| `adaptive_loss.py` | Part 2 — Parametric loss (Eq.6) |
| `state.py` | Part 3 — State representation |
| `controller.py` | Part 4a/4b — Policy + reward |
| `train_ala.py` | Part 4c — Training loop |
| `analysis.py` | Part 5a — Visualization |
| `answers.md` | Parts 2a, 2c, 2d — Written answers |
