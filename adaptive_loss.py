"""Part 2: Parametric adaptive loss function from ALA paper (Eq. 6).

l_Φ(f_w(x), y) = -σ(y^T Φ log f_w(y|x))

Φ is a learnable class-relationship matrix updated by an RL controller (Part 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLoss(nn.Module):
    """Adaptive loss with parametric class-relationship matrix Φ.

    Φ is not updated by gradient descent; it is adjusted externally
    by the RL controller via update_phi().
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Φ is not a trainable parameter — RL agent controls it
        self.phi = nn.Parameter(
            torch.eye(num_classes), requires_grad=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute adaptive loss.

        Args:
            logits: (B, C) raw model outputs before softmax.
            targets: (B,) integer class labels.

        Returns:
            Scalar loss (batch mean).
        """
        # log_softmax is numerically stable (avoids softmax → log roundtrip)
        log_probs = F.log_softmax(logits, dim=1)             # (B, C)
        y = F.one_hot(targets, self.num_classes).float()      # (B, C)
        weighted = y @ self.phi                               # (B, C)
        inner = (weighted * log_probs).sum(dim=1)             # (B,)
        # Clamp to keep sigmoid in a gradient-friendly range
        # sigmoid(-10) ≈ 4.5e-5, sigmoid(0) = 0.5
        inner = inner.clamp(-10.0, 0.0)
        sig = torch.sigmoid(inner)                            # (B,)
        return (-sig).mean()

    def update_phi(self, delta_phi: torch.Tensor) -> None:
        """Apply an additive update to Φ, then enforce constraints.

        Diagonal elements are kept at 1.0 (RL only adjusts off-diagonal
        class relationships). Off-diagonal values are clamped to [-1, 1]
        and symmetry is enforced.

        Args:
            delta_phi: (C, C) tensor of updates to add to Φ.
        """
        self.phi.data += delta_phi
        # Diagonal stays at 1 — RL controls off-diagonal only
        self.phi.data.fill_diagonal_(1.0)
        # Enforce symmetry: Φ(i,j) = Φ(j,i)
        self.phi.data = (self.phi.data + self.phi.data.T) / 2
        # Restore diagonal after symmetry averaging
        self.phi.data.fill_diagonal_(1.0)
        # Clamp off-diagonal only to [-1, 1]
        diag_mask = torch.eye(
            self.num_classes, device=self.phi.device, dtype=torch.bool,
        )
        self.phi.data[~diag_mask] = self.phi.data[~diag_mask].clamp(-1, 1)

    def reset_phi(self) -> None:
        """Reset Φ to the identity matrix."""
        self.phi.data = torch.eye(self.num_classes, device=self.phi.device)
