"""Export the SudokuDiT denoiser to ONNX for in-browser inference (onnxruntime-web).

    uv run python scripts/export_onnx.py                 # from the HF Hub model
    uv run python scripts/export_onnx.py --ckpt path.pt --num-heads 4

Exports a single forward pass ``logits = model(board, t)`` (the iterative decode loop
is reimplemented in the web app). Verifies the ONNX output matches PyTorch, then writes
to ``web/public/sudoku_dit.onnx``.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="local checkpoint (default: HF Hub model)")
    p.add_argument("--repo", default=None, help="HF repo id (default: nonet.hub.DEFAULT_REPO)")
    p.add_argument("--num-heads", type=int, default=4, help="heads for a local --ckpt")
    p.add_argument("--out", default="web/public/sudoku_dit.onnx")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    if args.ckpt:
        from nonet.hub import config_from_state_dict
        from nonet.model import SudokuDiT
        sd = torch.load(args.ckpt, map_location="cpu")
        model = SudokuDiT(**config_from_state_dict(sd, args.num_heads))
        model.load_state_dict(sd)
        model.eval()
    else:
        from nonet.hub import DEFAULT_REPO, load_pretrained
        model = load_pretrained(args.repo or DEFAULT_REPO, "cpu")

    # explicit-softmax attention exports far more portably than fused SDPA
    for blk in model.blocks:
        blk.attn.fused_attn = False

    board = torch.randint(0, 10, (1, 81), dtype=torch.long)
    t = torch.rand(1, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        model, (board, t), args.out,
        input_names=["board", "t"], output_names=["logits"],
        opset_version=args.opset, dynamo=False,
    )

    # parity check: PyTorch vs onnxruntime on a few random boards
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    max_err = 0.0
    with torch.no_grad():
        for _ in range(5):
            b = torch.randint(0, 10, (1, 81), dtype=torch.long)
            tt = torch.rand(1, dtype=torch.float32)
            ref = model(b, tt).numpy()
            got = sess.run(["logits"], {"board": b.numpy(), "t": tt.numpy()})[0]
            max_err = max(max_err, float(np.abs(ref - got).max()))
    sz = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({sz:.1f} MB, opset {args.opset})")
    print(f"max |pytorch - onnx| over 5 boards: {max_err:.2e}  -> {'OK' if max_err < 1e-3 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
