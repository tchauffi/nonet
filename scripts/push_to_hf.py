"""Publish a trained SudokuDiT checkpoint to the Hugging Face Hub.

    uv run python scripts/push_to_hf.py \
        --ckpt checkpoints/<run>/sudoku_dit_final.pt \
        --repo <user>/sudoku-dit --num-heads 4

Requires authentication first (run once):  uv run huggingface-cli login
Uploads weights as model.safetensors, a config.json, and a model card.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile

import torch

from nonet.hub import CONFIG_FILE, DEFAULT_REPO, WEIGHTS_FILE, config_from_state_dict

CARD = """---
library_name: nonet
pipeline_tag: other
tags: [sudoku, masked-diffusion, diffusion-transformer, dit]
---

# SudokuDiT — masked-diffusion Sudoku solver

A small Diffusion Transformer ({params:.2f}M params, `hidden={hidden_size}`,
`heads={num_heads}`, `blocks={num_blocks}`) trained as a discrete denoiser (MDLM-style)
that solves Sudoku by iteratively un-masking the most-confident cells. See the
[`nonet`](https://github.com/tchauffi/nonet) repository for training and the web demo.

## Usage

```python
from nonet.hub import load_solver
import torch

solver = load_solver("{repo}")                      # downloads weights + config
puzzle = torch.tensor([[int(c) for c in puzzle_str]])  # (1, 81), 0 = blank
solution = solver.solve(puzzle, conf_threshold=0.999)  # adaptive reveal
```

## Performance

Evaluated on held-out validation puzzles with the adaptive decoder (τ=0.999):
**99.8% valid solutions**, averaging ~3.6 reveal steps (easy boards in 1 step, the
hard 60+ blank tail in ~50). `exact_match` vs the reference is lower (~94%) only
because the dataset's high-blank puzzles aren't all uniquely solvable.

Input tokens: `0` = empty/`[MASK]`, `1..9` = digits, row-major over the 9×9 grid.
"""


def main():
    p = argparse.ArgumentParser(description="Push a SudokuDiT checkpoint to the HF Hub.")
    p.add_argument("--ckpt", default=None, help="checkpoint .pt (default: newest under checkpoints/)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="target Hub repo, e.g. user/sudoku-dit")
    p.add_argument("--num-heads", type=int, default=4, help="attention heads used at training")
    p.add_argument("--private", action="store_true", help="create a private repo")
    args = p.parse_args()

    ckpt = args.ckpt or max(glob.glob("checkpoints/**/*.pt", recursive=True), key=os.path.getmtime)
    sd = {k: v.contiguous() for k, v in torch.load(ckpt, map_location="cpu").items()}
    cfg = config_from_state_dict(sd, args.num_heads)
    params = sum(v.numel() for v in sd.values()) / 1e6
    print(f"checkpoint {ckpt}\nconfig {cfg}  ({params:.2f}M params)")

    from huggingface_hub import HfApi, create_repo
    from safetensors.torch import save_file

    create_repo(args.repo, exist_ok=True, private=args.private)
    api = HfApi()
    with tempfile.TemporaryDirectory() as d:
        save_file(sd, os.path.join(d, WEIGHTS_FILE))
        with open(os.path.join(d, CONFIG_FILE), "w") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(CARD.format(repo=args.repo, params=params, **cfg))
        api.upload_folder(repo_id=args.repo, folder_path=d, commit_message="Upload SudokuDiT")
    print(f"\n  ▸ pushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
