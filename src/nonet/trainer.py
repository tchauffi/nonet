import argparse
from datetime import datetime
from pathlib import Path

import accelerate
import torch
from accelerate.utils import set_seed
from torch.optim import AdamW

from nonet.dataset import get_train_dataloader
from nonet.model import SudokuDiT
from nonet.pipeline import MASK_TOKEN, SudokuSolver
from nonet.schedueler import LinearScheduler
from nonet.sudokuer import SudokuJudge


def parse_args():
    p = argparse.ArgumentParser(description="Train the SudokuDiT masked-diffusion solver.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5_000)
    p.add_argument("--solve-steps", type=int, default=27)
    p.add_argument("--seed", type=int, default=42)
    cond = p.add_mutually_exclusive_group()
    cond.add_argument("--conditional", dest="conditional", action="store_true",
                      help="keep the puzzle's clues unmasked during training (default)")
    cond.add_argument("--unconditional", dest="conditional", action="store_false",
                      help="mask anywhere in the solution; learn the full generative model")
    p.set_defaults(conditional=True)
    # model size
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=12)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--run-dir", type=str, default="runs")
    return p.parse_args()


def lr_lambda(warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    return fn


@torch.no_grad()
def evaluate(solver, judge, val_iter, dataloader, n_batches, solve_steps, device):
    """Solve a few val puzzles and report cell accuracy + solved rate."""
    solver.model.eval()
    n_cells = n_solved = n_total = correct = 0
    for _ in range(n_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(dataloader)
            batch = next(val_iter)
        puzzles = batch["puzzle"].to(device)
        solutions = batch["solution"].to(device)

        pred = solver.solve(puzzles, num_steps=solve_steps)
        blanks = puzzles == MASK_TOKEN
        correct += ((pred == solutions) & blanks).sum().item()
        n_cells += blanks.sum().item()
        n_solved += judge.is_solved(pred).sum().item()
        n_total += puzzles.shape[0]

    solver.model.train()
    cell_acc = correct / max(n_cells, 1)
    solved_rate = n_solved / max(n_total, 1)
    return {"eval/cell_acc": cell_acc, "eval/solved_rate": solved_rate}, val_iter


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    accelerator = accelerate.Accelerator(log_with="tensorboard", project_dir=args.run_dir)
    accelerator.init_trackers(run_name)
    accelerator.print(f"run: {run_name}")
    device = accelerator.device

    model = SudokuDiT(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        num_blocks=args.num_blocks,
        grid_size=9,
    )
    solver = SudokuSolver(model, LinearScheduler())
    judge = SudokuJudge()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args.warmup_steps))

    train_dl, val_dl = get_train_dataloader(batch_size=args.batch_size, seed=args.seed)

    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)
    solver.model = model  # use the prepared (possibly DDP-wrapped) model

    ckpt_dir = Path(args.ckpt_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    val_iter = iter(val_dl)
    step = 0
    train_iter = iter(train_dl)
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        puzzles = batch["puzzle"]
        solutions = batch["solution"]
        # conditional: keep the puzzle's clues unmasked (never masked/scored).
        # unconditional: mask anywhere in the solution (full generative model).
        keep = (puzzles != MASK_TOKEN) if args.conditional else None

        with accelerator.accumulate(model):
            loss = solver.loss(solutions, keep=keep)
            accelerator.backward(loss)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()

        if step % args.log_every == 0:
            loss_val = accelerator.gather(loss.detach()).mean().item()
            accelerator.log(
                {"train/loss": loss_val, "train/lr": lr_sched.get_last_lr()[0]}, step=step
            )
            accelerator.print(f"step {step:>7d} | loss {loss_val:.4f}")

        if step > 0 and step % args.eval_every == 0:
            metrics, val_iter = evaluate(
                solver, judge, val_iter, val_dl, args.eval_batches, args.solve_steps, device
            )
            accelerator.log(metrics, step=step)
            accelerator.print(
                f"step {step:>7d} | cell_acc {metrics['eval/cell_acc']:.3f} "
                f"| solved {metrics['eval/solved_rate']:.3f}"
            )

        if step > 0 and step % args.save_every == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), ckpt_dir / f"sudoku_dit_{step}.pt")

        step += 1

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), ckpt_dir / "sudoku_dit_final.pt")
    accelerator.end_training()


if __name__ == "__main__":
    main()
