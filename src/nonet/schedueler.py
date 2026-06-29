import math

import torch
from torch import nn


class LinearScheduler(nn.Module):
    """Linear (log-linear absorbing) masking schedule for MDLM.

    Continuous diffusion time ``t in [0, 1]``:

        alpha_t      = 1 - t     probability a token is *kept* (clean)
        1 - alpha_t  = t         probability a token is [MASK]

    ``alpha_t`` decreases from ~1 (clean data) at ``t = 0`` to ~0 (everything
    masked) at ``t = 1``. Sahoo et al. (2024, arXiv:2406.07524) show the
    continuous-time NELBO is invariant to the choice of noise schedule, so this
    simplest schedule is used by default. ``d_alpha`` is its time derivative,
    needed to weight the training loss.
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def d_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t)


class CosineScheduler(nn.Module):
    """Cosine (MaskGIT-style) masking schedule.

        alpha_t     = cos(pi/2 * t)     probability a token is *kept* (clean)
        1 - alpha_t = 1 - cos(pi/2 * t) probability a token is [MASK]

    Like :class:`LinearScheduler` it runs ``alpha`` from 1 (t=0, clean) to 0 (t=1,
    fully masked), but the cosine bends the curve so the schedule spends *more* of its
    range at low mask ratios -- i.e. training samples more near-complete boards and
    fewer almost-empty ones. ``d_alpha`` is its time derivative (unused by the current
    unweighted loss, kept for parity with the schedule interface / weighted NELBO).
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * math.pi * t)

    def d_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * math.pi * torch.sin(0.5 * math.pi * t)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t)
