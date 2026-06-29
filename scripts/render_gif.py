"""Render a SudokuDiT solve as an animated GIF (mirrors the web UI's heatmap).

    uv run python scripts/render_gif.py --mode confidence --tau 0.999 \
        --min-blank 44 --max-blank 50 --out assets/confidence_solve.gif

Each frame is a 9x9 board: given clues in light grey, model-filled cells tinted
by the model's step-0 confidence (dim blue = unsure -> green = sure), cells that
disagree with the reference solution outlined in red.
"""
from __future__ import annotations

import argparse
import colorsys
import glob
import os
import random

import torch
from PIL import Image, ImageDraw, ImageFont

from nonet.model import SudokuDiT
from nonet.pipeline import MASK_TOKEN, SudokuSolver
from nonet.schedueler import LinearScheduler
from nonet.sudokuer import SudokuJudge

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = (17, 19, 26)
EMPTY = (24, 27, 36)
CLUE_BG, CLUE_FG = (31, 35, 48), (244, 246, 251)
FILL_FG = (238, 241, 248)
THIN, BOLD = (42, 47, 61), (91, 100, 120)
BAD = (239, 106, 106)


def heat(conf: float):
    """Confidence -> cool blue (unsure) .. green (sure) tint, matching the UI.

    Deliberately avoids red, which is reserved for cells that disagree with the
    reference solution (so 'unsure' never reads as 'wrong').
    """
    n = max(0.0, min(1.0, (conf - 0.6) / 0.4))
    h = (222 - 72 * n) / 360.0          # 222deg blue -> 150deg green
    r, g, b = colorsys.hls_to_rgb(h, 0.26 + 0.06 * n, 0.50 + 0.08 * n)
    return (int(r * 255), int(g * 255), int(b * 255))


def load_solver(ckpt, num_heads, device):
    sd = torch.load(ckpt, map_location=device)
    hidden = sd["square_embed.weight"].shape[1]
    grid = sd["square_embed.weight"].shape[0] - 1
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    mlp_ratio = sd["blocks.0.mlp.0.weight"].shape[0] / hidden
    m = SudokuDiT(hidden_size=hidden, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  num_blocks=n_blocks, grid_size=grid).to(device)
    m.load_state_dict(sd); m.eval()
    return SudokuSolver(m, LinearScheduler())


def render_frame(grid, clues, conf, solution, cell, font, foot_h, font_s, caption):
    n = 9
    W = n * cell
    img = Image.new("RGB", (W, W + foot_h), BG)
    d = ImageDraw.Draw(img)
    for i in range(81):
        r, c = i // n, i % n
        x0, y0 = c * cell, r * cell
        v = grid[i]
        if v == 0:
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=EMPTY)
            continue
        if clues[i]:
            bg, fg = CLUE_BG, CLUE_FG
        else:
            bg, fg = heat(conf[i]), FILL_FG
        d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=bg)
        if solution and not clues[i] and solution[i] and v != solution[i]:
            d.rectangle([x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2], outline=BAD, width=2)
        tb = d.textbbox((0, 0), str(v), font=font)
        d.text((x0 + (cell - (tb[2] - tb[0])) / 2, y0 + (cell - (tb[3] - tb[1])) / 2 - tb[1]),
               str(v), font=font, fill=fg)
    # grid rules
    for k in range(n + 1):
        w = 3 if k % 3 == 0 else 1
        col = BOLD if k % 3 == 0 else THIN
        d.line([(k * cell, 0), (k * cell, W)], fill=col, width=w)
        d.line([(0, k * cell), (W, k * cell)], fill=col, width=w)
    if caption:
        d.text((8, W + 7), caption, font=font_s, fill=(139, 146, 166))
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--mode", choices=["confidence", "iterative"], default="confidence")
    p.add_argument("--tau", type=float, default=0.999)
    p.add_argument("--steps", type=int, default=27)
    p.add_argument("--min-blank", type=int, default=44)
    p.add_argument("--max-blank", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cell", type=int, default=46)
    p.add_argument("--ms", type=int, default=180, help="ms per step")
    p.add_argument("--hold", type=int, default=2000, help="ms to hold the final frame")
    p.add_argument("--out", default="assets/solve.gif")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or max(glob.glob("checkpoints/**/*.pt", recursive=True), key=os.path.getmtime)
    solver = load_solver(ckpt, args.num_heads, device)
    judge = SudokuJudge()

    from datasets import load_dataset
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="validation")
    random.seed(args.seed)
    kw = dict(return_trace=True, constraint_tiebreak=False)
    kw.update(conf_threshold=args.tau) if args.mode == "confidence" else kw.update(num_steps=81)
    if args.mode == "iterative":
        kw["num_steps"] = args.steps
    # pick the first puzzle in range that the chosen mode actually solves
    puzzle = solution = trace = conf = None
    for i in random.sample(range(len(ds)), 4000):
        row = ds[i]; pz = [int(c) for c in row["puzzle"]]
        if not (args.min_blank <= pz.count(0) <= args.max_blank):
            continue
        x = torch.tensor(pz, device=device).unsqueeze(0)
        _, tr, ic = solver.solve(x, **kw)
        if judge.is_solved(tr[-1].to(device))[0].item():
            puzzle, solution, trace, conf = pz, [int(c) for c in row["solution"]], tr, ic[0].tolist()
            break
    if trace is None:
        raise SystemExit("no solvable puzzle found in range; widen --min/max-blank")

    clues = [v != MASK_TOKEN for v in puzzle]
    font = ImageFont.truetype(FONT, int(args.cell * 0.55))
    foot_h = 30
    font_s = ImageFont.truetype(FONT, 13)
    how = f"τ={args.tau:g}" if args.mode == "confidence" else f"{args.steps} steps"
    frames, durations = [], []
    for s, snap in enumerate(trace):
        g = snap[0].tolist()
        last = s == len(trace) - 1
        cap = f"{puzzle.count(0)} blanks · {how} · step {s}/{len(trace)-1}"
        if last:
            cap += " · ✓ solved" if judge.is_solved(snap.to(device))[0].item() else " · ✗ conflicts"
        frames.append(render_frame(g, clues, conf, solution, args.cell, font, foot_h, font_s, cap))
        durations.append(args.hold if last else args.ms)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    print(f"wrote {args.out}  ({len(frames)} frames, {puzzle.count(0)} blanks)")


if __name__ == "__main__":
    main()
