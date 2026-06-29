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
import shutil
import tempfile

import torch

from nonet.hub import CONFIG_FILE, DEFAULT_REPO, WEIGHTS_FILE, config_from_state_dict

CARD = """---
license: apache-2.0
pipeline_tag: other
library_name: pytorch
tags:
  - sudoku
  - masked-diffusion
  - diffusion-transformer
  - dit
datasets:
  - Ritvik19/Sudoku-Dataset
---

# SudokuDiT

A compact **Diffusion Transformer (NPARAMS M params)** that solves Sudoku as a
**masked discrete diffusion** (MDLM-style) denoiser: it fills a puzzle by iteratively
un-masking the cells it is most confident about, MaskGIT-style.

![watch it solve](solve.gif)

- **Architecture:** DiT with adaLN-Zero conditioning — `hidden=HIDDEN`, `heads=HEADS`,
  `blocks=BLOCKS`; per-cell token + 2-D positional + 3x3-box embeddings, plus a timestep.
- **Code, training and an interactive web demo:** <https://github.com/tchauffi/nonet>

## Input / output

A board is **81 tokens, row-major**: `0` = empty / `[MASK]`, `1..9` = digits. The solver
clamps the given clues and only fills the blanks.

## Usage

```python
import torch
from nonet.hub import load_solver   # pip install git+https://github.com/tchauffi/nonet

solver = load_solver("REPO_ID")     # downloads model.safetensors + config.json
puzzle = "417000800030005900800000000050000600000700020000000000000060054000200000000000003"
x = torch.tensor([[int(c) for c in puzzle]])      # (1, 81)
solution = solver.solve(x, conf_threshold=0.999)  # adaptive reveal (recommended)
# fixed-budget alternative: solver.solve(x, num_steps=81)
```

## Performance

Held-out validation puzzles, adaptive decoder (tau = 0.999):

| metric | value |
|--------|-------|
| valid solutions | **99.8 %** |
| exact match vs reference | 94.3 % |
| avg reveal steps / puzzle | 3.6 |

Solve rate is 100 % up to ~39 blanks and ~98 % on the hard 50+ blank tail (where the adaptive
decoder spends more steps). `exact_match` is lower than `valid` only because the dataset's
high-blank puzzles aren't always uniquely solvable, so the model may return a *different*
valid grid.

## Training

- Data: [`Ritvik19/Sudoku-Dataset`](https://huggingface.co/datasets/Ritvik19/Sudoku-Dataset)
  (~17 M puzzles), tokenized on the fly.
- Objective: masked cross-entropy over masked cells, conditional (given clues are never
  masked), with a linear masking schedule.
- ~50 k steps, AdamW, batch 256. Kept deliberately small — larger models sit at the entropy
  floor far longer before the loss breaks through.

## Limitations

- Not a guaranteed solver: a small fraction of very hard (60+ blank) boards come out invalid.
- Trained only on standard 9x9 Sudoku from the dataset above.
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
    card = (CARD.replace("REPO_ID", args.repo).replace("NPARAMS", f"{params:.2f}")
                .replace("HIDDEN", str(cfg["hidden_size"]))
                .replace("HEADS", str(cfg["num_heads"]))
                .replace("BLOCKS", str(cfg["num_blocks"])))
    with tempfile.TemporaryDirectory() as d:
        save_file(sd, os.path.join(d, WEIGHTS_FILE))
        with open(os.path.join(d, CONFIG_FILE), "w") as f:
            json.dump(cfg, f, indent=2)
        gif = "assets/iterative_solve.gif"
        if os.path.exists(gif):
            shutil.copyfile(gif, os.path.join(d, "solve.gif"))
        else:
            card = card.replace("![watch it solve](solve.gif)\n\n", "")
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(card)
        api.upload_folder(repo_id=args.repo, folder_path=d, commit_message="Update SudokuDiT card + assets")
    print(f"\n  ▸ pushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
