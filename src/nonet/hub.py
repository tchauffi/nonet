"""Load and publish SudokuDiT weights on the Hugging Face Hub.

The Hub repo holds the weights as ``model.safetensors`` plus a ``config.json`` with
the architecture dims (``num_heads`` in particular is not recoverable from the
weights alone). Load a published model with :func:`load_pretrained` /
:func:`load_solver`; publish a local checkpoint with ``scripts/push_to_hf.py``.
"""
from __future__ import annotations

import json

import torch

from nonet.model import SudokuDiT
from nonet.pipeline import SudokuSolver
from nonet.schedueler import LinearScheduler

DEFAULT_REPO = "tchauffi/sudoku-dit"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


def config_from_state_dict(sd: dict, num_heads: int) -> dict:
    """Infer SudokuDiT constructor kwargs from a state dict (+ num_heads, which the
    weights don't encode)."""
    hidden = sd["square_embed.weight"].shape[1]
    return {
        "hidden_size": hidden,
        "num_heads": num_heads,
        "mlp_ratio": sd["blocks.0.mlp.0.weight"].shape[0] / hidden,
        "num_blocks": len({k.split(".")[1] for k in sd if k.startswith("blocks.")}),
        "grid_size": sd["square_embed.weight"].shape[0] - 1,  # vocab = digits + MASK
    }


def load_pretrained(repo_id: str = DEFAULT_REPO, device: str = "cpu") -> SudokuDiT:
    """Download a published SudokuDiT from the Hub and return it in eval mode."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    with open(hf_hub_download(repo_id, CONFIG_FILE)) as f:
        cfg = json.load(f)
    model = SudokuDiT(**cfg).to(device)
    model.load_state_dict(load_file(hf_hub_download(repo_id, WEIGHTS_FILE), device=device))
    model.eval()
    return model


def load_solver(repo_id: str = DEFAULT_REPO, device: str = "cpu") -> SudokuSolver:
    """Convenience: a ready-to-use SudokuSolver around the published model."""
    return SudokuSolver(load_pretrained(repo_id, device), LinearScheduler())
