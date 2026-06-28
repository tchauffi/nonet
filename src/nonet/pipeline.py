import torch
import torch.nn.functional as F
from torch import nn

MASK_TOKEN = 0   # absorbing [MASK] / empty cell
NUM_DIGITS = 9   # clean classes: digits 1..9


class SudokuSolver(nn.Module):
    def __init__(self, model: nn.Module, scheduler: nn.Module, eps: float = 1e-3):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.eps = eps  # keeps t in (eps, 1-eps), avoiding the 1/t weight blow-up

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """Sample continuous diffusion times t in (eps, 1 - eps) for a batch."""
        device = next(self.model.parameters()).device
        t = torch.rand(batch_size, device=device)
        return (1.0 - 2.0 * self.eps) * t + self.eps

    def q_sample(self, x: torch.Tensor, t: torch.Tensor,
                 keep: torch.Tensor | None = None) -> torch.Tensor:
        """Sample from the forward diffusion process q(x_t | x_0).

        ``keep`` is an optional (B, 81) bool of cells to never mask (e.g. a
        puzzle's given clues used as conditioning).
        """
        alpha_t = self.scheduler.alpha(t)

        mask_prob = 1 - alpha_t
        masked = torch.rand_like(x.float()) < mask_prob.unsqueeze(-1)  # Randomly mask tokens based on the masking probability
        if keep is not None:
            masked = masked & ~keep

        z_t = torch.where(masked, torch.full_like(x, MASK_TOKEN), x)  # Replace masked tokens with [MASK]
        return z_t, masked

    def loss(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """Weighted masked cross-entropy (continuous-time NELBO, Sahoo et al. 2024).

        L = E_t [ -alpha'_t / (1 - alpha_t) * sum_{masked cells} log p_theta(x | z_t) ]

        ``x`` is the clean solution (B, 81) with digits 1..9. ``keep`` marks
        conditioning cells (given clues) that are never masked and never scored.
        Returned value is normalised per cell so it stays O(1).
        """
        t = self.sample_t(x.shape[0])
        z_t, masked = self.q_sample(x, t, keep=keep)

        logits = self.model(z_t, t)                                   # (B, 81, 9)

        # model classes 0..8 map to digits 1..9, so the target is x - 1.
        # F.cross_entropy expects raw logits (it applies log_softmax itself).
        loss = F.cross_entropy(logits.permute(0, 2, 1), x - 1, reduction="none")  # (B, 81)
        loss = loss * masked.float()  # only score masked cells
        loss = loss.sum(dim=1) / masked.sum(dim=1).clamp(min=1.0)  # normalise per cell

        return loss.mean() 
        

    @torch.no_grad()
    def sample(self, problem: torch.Tensor, num_steps: int = 10,
               temperature: float = 1.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample a solution for a given Sudoku problem using the reverse diffusion process.

        ``problem`` is (B, 81) with ``MASK_TOKEN`` (0) in the blanks to fill.
        The given clues (non-zero cells) are clamped via ``keep`` and never
        overwritten. We walk diffusion time from t=1 down to t=0, revealing a
        still-masked cell at step ``t -> s`` with probability
        ``(alpha_s - alpha_t) / (1 - alpha_t)`` and drawing its digit from the
        denoiser's prediction; revealed cells are carried over unchanged.
        """
        device = next(self.model.parameters()).device
        x_t = problem.clone().to(device)
        keep = x_t != MASK_TOKEN          # given clues: never overwrite
        B, L = x_t.shape

        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for step in range(num_steps):
            t = ts[step].expand(B)
            alpha_t = self.scheduler.alpha(t).view(B, 1)
            alpha_s = self.scheduler.alpha(ts[step + 1].expand(B)).view(B, 1)

            logits = self.model(x_t, t) / temperature              # (B, L, 9)
            probs = F.softmax(logits, dim=-1).reshape(-1, NUM_DIGITS)
            sampled = torch.multinomial(probs, 1, generator=generator).view(B, L) + 1  # class -> digit

            unmask_prob = (alpha_s - alpha_t) / (1.0 - alpha_t)    # (B, 1)
            is_mask = x_t == MASK_TOKEN
            do_unmask = is_mask & ~keep & (
                torch.rand(B, L, device=device, generator=generator) < unmask_prob
            )
            x_t = torch.where(do_unmask, sampled, x_t)

        # t -> 0: greedily fill anything still masked
        leftover = (x_t == MASK_TOKEN) & ~keep
        if leftover.any():
            fill = self.model(x_t, ts[-1].expand(B)).argmax(dim=-1) + 1
            x_t = torch.where(leftover, fill, x_t)
        return x_t

    @torch.no_grad()
    def solve(self, problem: torch.Tensor, num_steps: int = 27,
              temperature: float = 0.0,
              generator: torch.Generator | None = None) -> torch.Tensor:
        """Solve a puzzle by revealing the most-confident cells first (MaskGIT-style).

        Unlike :meth:`sample` (random reveal order), this fills, at each step,
        only the highest-confidence still-masked cells -- the right decoder for a
        unique-solution constraint puzzle: low-confidence cells stay open until
        context pins them down.

        ``problem`` is (B, 81) with ``MASK_TOKEN`` (0) in the blanks to fill;
        non-zero clues are clamped and never overwritten. ``temperature=0`` is
        greedy argmax; >0 samples the per-cell digit but still *orders* reveals
        by confidence.
        """
        device = next(self.model.parameters()).device
        x_t = problem.clone().to(device)
        keep = x_t != MASK_TOKEN              # given clues: never overwrite
        B, L = x_t.shape
        n_blank = (~keep).sum(dim=1)          # (B,) cells to fill per board

        ts = torch.linspace(1.0, 0.0, num_steps, device=device)
        for step in range(num_steps):
            is_mask = x_t == MASK_TOKEN
            if not is_mask.any():
                break

            logits = self.model(x_t, ts[step].expand(B))          # (B, L, 9)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                pred = torch.multinomial(probs.reshape(-1, NUM_DIGITS), 1,
                                         generator=generator).view(B, L) + 1  # class -> digit
                conf = probs.max(dim=-1).values
            else:
                conf, pred = logits.softmax(dim=-1).max(dim=-1)
                pred = pred + 1                                # class -> digit
            conf = conf.masked_fill(~is_mask, -1.0)               # only rank masked cells

            # how many cells should be revealed by the end of this step
            revealed_target = torch.ceil(n_blank * (step + 1) / num_steps).long()
            already = n_blank - is_mask.sum(dim=1)
            k = (revealed_target - already).clamp(min=0)          # reveal this step (B,)
            if step == num_steps - 1:
                k = is_mask.sum(dim=1)                             # finish anything left

            rank = conf.argsort(dim=1, descending=True).argsort(dim=1)
            reveal = (rank < k.unsqueeze(1)) & is_mask
            x_t = torch.where(reveal, pred, x_t)

        return x_t


if __name__ == "__main__":
    from nonet.model import SudokuDiT
    from nonet.schedueler import LinearScheduler

    # Example usage
    model = SudokuDiT(hidden_size=512, num_heads=8, mlp_ratio=4.0, num_blocks=12, grid_size=9)
    scheduler = LinearScheduler()
    solver = SudokuSolver(model, scheduler)

    # Sample a batch of puzzles (for demonstration purposes)
    batch_size = 4
    puzzles = torch.randint(1, 10, (batch_size, 81))  # Random Sudoku puzzles

    # Sample diffusion times
    t = solver.sample_t(batch_size)

    for t in torch.linspace(0, 1, steps=5):  # Example diffusion times
        z_t, masked = solver.q_sample(puzzles, t)
        print(f"Diffusion time: {t.item():.2f}, Masked puzzles:\n{z_t}\nMasked positions:\n{masked}\n")    
