// In-browser SudokuDiT: ONNX forward pass + the confidence-first decode loop,
// ported from src/nonet/pipeline.py::SudokuSolver.solve (batch size 1, greedy).
//
// The denoiser is a single forward `logits = model(board, t)`; the iterative
// MaskGIT-style reveal is reimplemented here so the whole solve runs client-side.

import type { InferenceSession, Tensor } from 'onnxruntime-web';

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || '';
const N = 81;
const D = 9; // digit classes 1..9

let sessionPromise: Promise<InferenceSession> | null = null;

/** Load (once) the onnxruntime-web session. Dynamic import keeps ORT out of SSR. */
export function loadModel(): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import('onnxruntime-web');
      // GitHub Pages isn't cross-origin isolated, so no SharedArrayBuffer/threads.
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths =
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
      return ort.InferenceSession.create(`${BASE}/sudoku_dit.onnx`, {
        executionProviders: ['wasm'],
      });
    })();
  }
  return sessionPromise;
}

/** One denoiser pass: board (81 digits, 0=blank) + time t -> per-cell logits (81x9). */
async function forward(
  session: InferenceSession,
  board: number[],
  t: number,
): Promise<Float32Array> {
  const ort = await import('onnxruntime-web');
  const boardT: Tensor = new ort.Tensor(
    'int64',
    BigInt64Array.from(board, (v) => BigInt(v)),
    [1, N],
  );
  const tT: Tensor = new ort.Tensor('float32', Float32Array.from([t]), [1]);
  const out = await session.run({ board: boardT, t: tT });
  return out.logits.data as Float32Array;
}

/** Softmax over a cell's 9 logits -> {digit 1..9 argmax, max probability}. */
function cellPred(logits: Float32Array, i: number): { digit: number; conf: number } {
  let max = -Infinity;
  let arg = 0;
  for (let d = 0; d < D; d++) {
    const v = logits[i * D + d];
    if (v > max) {
      max = v;
      arg = d;
    }
  }
  let sum = 0;
  for (let d = 0; d < D; d++) sum += Math.exp(logits[i * D + d] - max);
  return { digit: arg + 1, conf: 1 / sum };
}

export type Frame = {
  grid: number[]; // 81 digits, 0 = still blank
  conf: number[]; // step-0 confidence per cell, 0 for clues (drives tinting)
  clue: boolean[]; // given cells, never overwritten
  step: number; // reveal steps taken so far
  solved: boolean; // grid is a complete legal Sudoku
  done: boolean; // decode loop finished
};

export type SolveMode = 'confidence' | 'iterative';

/**
 * Decode generator. Yields a frame after the initial board and after every
 * reveal step. `confidence` mode reveals every still-masked cell the model is
 * >= tau sure of (adaptive count); `iterative` mode follows the fixed linear
 * schedule for `steps` reveals.
 */
export async function* solve(
  session: InferenceSession,
  puzzle: number[],
  opts: { mode: SolveMode; tau: number; steps: number },
): AsyncGenerator<Frame> {
  const { mode, tau } = opts;
  const grid = [...puzzle];
  const clue = puzzle.map((v) => v !== 0);
  const nBlank = clue.filter((c) => !c).length;
  const maxSteps = mode === 'confidence' ? N : opts.steps;
  let initConf = new Array(N).fill(0);

  yield { grid: [...grid], conf: initConf, clue, step: 0, solved: false, done: false };
  if (nBlank === 0) {
    yield { grid: [...grid], conf: initConf, clue, step: 0, solved: isSolved(grid), done: true };
    return;
  }

  for (let step = 0; step < maxSteps; step++) {
    const isMask = grid.map((v) => v === 0);
    const numMask = isMask.reduce((a, m) => a + (m ? 1 : 0), 0);
    if (numMask === 0) break;

    // linear schedule: t = masked fraction (adaptive) or the fixed timestep grid.
    const t =
      mode === 'confidence'
        ? Math.min(Math.max(numMask / nBlank, 0), 1)
        : 1 - step / maxSteps;
    const logits = await forward(session, grid, t);

    const conf = new Array(N).fill(-1);
    const pred = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      if (!isMask[i]) continue; // only rank/fill masked cells
      const { digit, conf: c } = cellPred(logits, i);
      conf[i] = c;
      pred[i] = digit;
    }
    if (step === 0) initConf = conf.map((c) => Math.max(c, 0)); // clues (-1) -> 0

    // how many cells to reveal this step
    let k: number;
    if (mode === 'confidence') {
      k = 0;
      for (let i = 0; i < N; i++) if (isMask[i] && conf[i] >= tau) k++;
      k = Math.max(k, 1); // always make progress
    } else {
      const alphaNext = (step + 1) / maxSteps; // 1 - ts[step+1], linear
      const revealedTarget = Math.ceil(nBlank * alphaNext);
      const already = nBlank - numMask;
      k = Math.max(revealedTarget - already, 0);
    }
    if (step === maxSteps - 1) k = numMask; // finish anything left

    // reveal the k most-confident still-masked cells
    const order = grid
      .map((_, i) => i)
      .filter((i) => isMask[i])
      .sort((a, b) => conf[b] - conf[a]);
    for (let j = 0; j < k && j < order.length; j++) grid[order[j]] = pred[order[j]];

    const done = grid.every((v) => v !== 0) || step === maxSteps - 1;
    yield {
      grid: [...grid],
      conf: initConf,
      clue,
      step: step + 1,
      solved: done && isSolved(grid),
      done,
    };
    if (done) return;
  }
}

/** TS port of SudokuJudge.is_solved: a complete board with 1..9 in every unit. */
export function isSolved(grid: number[]): boolean {
  if (grid.some((v) => v < 1 || v > 9)) return false;
  const full = (vals: number[]) => {
    let seen = 0;
    for (const v of vals) seen |= 1 << v;
    return seen === 0b1111111110; // bits 1..9 set
  };
  for (let i = 0; i < 9; i++) {
    const row: number[] = [];
    const col: number[] = [];
    const box: number[] = [];
    for (let j = 0; j < 9; j++) {
      row.push(grid[i * 9 + j]);
      col.push(grid[j * 9 + i]);
      const br = Math.floor(i / 3) * 3 + Math.floor(j / 3);
      const bc = (i % 3) * 3 + (j % 3);
      box.push(grid[br * 9 + bc]);
    }
    if (!full(row) || !full(col) || !full(box)) return false;
  }
  return true;
}
