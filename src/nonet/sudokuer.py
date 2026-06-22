import torch
import torch.nn as nn


class SudokuJudge(nn.Module):
    """Vectorized Sudoku validator. Accepts flat (81,) or (B, 81) tensors."""

    def _normalize(self, board: torch.Tensor) -> torch.Tensor:
        if board.dim() == 1:
            board = board.unsqueeze(0)
        return board.view(-1, 9, 9)

    def _get_units(self, board: torch.Tensor) -> torch.Tensor:
        # board: (B, 9, 9) -> (B, 27, 9): rows + cols + boxes
        B = board.shape[0]
        rows = board
        cols = board.transpose(1, 2)
        boxes = board.view(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(B, 9, 9)
        return torch.cat([rows, cols, boxes], dim=1)

    def _digit_counts(self, units: torch.Tensor) -> torch.Tensor:
        # units: (B, 27, 9) with values 0-9
        # returns counts per digit 1-9: (B, 27, 9)
        B = units.shape[0]
        counts = torch.zeros(B, 27, 10, dtype=torch.long, device=units.device)
        counts.scatter_add_(2, units, torch.ones_like(units))
        return counts[:, :, 1:]

    def is_valid(self, board: torch.Tensor) -> torch.Tensor:
        counts = self._digit_counts(self._get_units(self._normalize(board)))
        return (counts <= 1).all(dim=(1, 2))

    def is_solved(self, board: torch.Tensor) -> torch.Tensor:
        counts = self._digit_counts(self._get_units(self._normalize(board)))
        return (counts == 1).all(dim=(1, 2))
