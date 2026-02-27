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
        # 1. softmax → probabilities
        probs = F.softmax(logits, dim=1)                    # (B, C)
        # 2. log probabilities with numerical stability
        log_probs = torch.log(probs + 1e-8)                 # (B, C)
        # 3. one-hot encoding
        y = F.one_hot(targets, self.num_classes).float()     # (B, C)
        # 4. weighted = y @ Φ  (selects the Φ row for each sample's class)
        weighted = y @ self.phi                              # (B, C)
        # 5. inner product: (weighted * log_probs).sum(dim=1)
        inner = (weighted * log_probs).sum(dim=1)            # (B,)
        # 6. sigmoid
        sig = torch.sigmoid(inner)                           # (B,)
        # 7. negative mean
        return (-sig).mean()

    def update_phi(self, delta_phi: torch.Tensor) -> None:
        """Apply an additive update to Φ, then enforce symmetry and clamp.

        Args:
            delta_phi: (C, C) tensor of updates to add to Φ.
        """
        self.phi.data += delta_phi
        # Enforce symmetry: Φ(i,j) = Φ(j,i)
        self.phi.data = (self.phi.data + self.phi.data.T) / 2
        # Clamp to [-1, 1]
        self.phi.data.clamp_(-1, 1)

    def reset_phi(self) -> None:
        """Reset Φ to the identity matrix."""
        self.phi.data = torch.eye(self.num_classes, device=self.phi.device)
