'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { loadModel, solve, type Frame, type SolveMode } from '@/lib/solver';

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || '';
const N = 81;

type Puzzle = { puzzle: string; solution: string };

// blue (unsure) -> green (sure); red is never used here so it can't read as "wrong"
function tint(p: number): string {
  const n = Math.min(Math.max((p - 0.6) / 0.4, 0), 1);
  return `hsl(${222 - 72 * n} ${50 + 8 * n}% ${26 + 6 * n}%)`;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Parse a pasted board into 81 ints (0 = blank). Lenient about formatting:
// keeps 1-9 as digits and 0/./_/- as blanks, ignores all other chars
// (whitespace, commas, pipes, newlines). Returns null unless exactly 81 cells.
function parsePuzzle(text: string): number[] | null {
  const cells: number[] = [];
  for (const ch of text) {
    if (ch >= '1' && ch <= '9') cells.push(ch.charCodeAt(0) - 48);
    else if (ch === '0' || ch === '.' || ch === '_' || ch === '-') cells.push(0);
  }
  return cells.length === N ? cells : null;
}

// Are the given (non-zero) clues internally consistent — no repeated digit in
// any row, column, or 3x3 box? (A blank board is trivially valid.)
function cluesValid(grid: number[]): boolean {
  const noDup = (idxs: number[]) => {
    const seen = new Set<number>();
    for (const i of idxs) {
      const v = grid[i];
      if (v === 0) continue;
      if (seen.has(v)) return false;
      seen.add(v);
    }
    return true;
  };
  for (let u = 0; u < 9; u++) {
    const row: number[] = [];
    const col: number[] = [];
    const box: number[] = [];
    for (let j = 0; j < 9; j++) {
      row.push(u * 9 + j);
      col.push(j * 9 + u);
      const br = Math.floor(u / 3) * 3 + Math.floor(j / 3);
      const bc = (u % 3) * 3 + (j % 3);
      box.push(br * 9 + bc);
    }
    if (!noDup(row) || !noDup(col) || !noDup(box)) return false;
  }
  return true;
}

export default function Page() {
  const [puzzles, setPuzzles] = useState<Puzzle[]>([]);
  const [puzzle, setPuzzle] = useState<number[]>(() => new Array(N).fill(0)); // input grid
  const [frame, setFrame] = useState<Frame | null>(null); // solve output (null = editing)
  const [selected, setSelected] = useState<number | null>(null);
  const [mode, setMode] = useState<SolveMode>('confidence');
  const [tau1000, setTau1000] = useState(999);
  const [steps, setSteps] = useState(27);
  const [delay, setDelay] = useState(140);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('loading model…');
  const [ready, setReady] = useState(false);
  const [pasteText, setPasteText] = useState('');

  const runId = useRef(0);
  const delayRef = useRef(delay);
  delayRef.current = delay;
  const boardRef = useRef<HTMLDivElement>(null);

  // fetch the bundled puzzle pool + warm up the ONNX session
  useEffect(() => {
    let alive = true;
    fetch(`${BASE}/puzzles.json`)
      .then((r) => r.json())
      .then((p: Puzzle[]) => {
        if (!alive) return;
        setPuzzles(p);
        const i = Math.floor(Math.random() * p.length);
        setPuzzle(p[i].puzzle.split('').map(Number));
      });
    loadModel()
      .then(() => alive && (setReady(true), setStatus('ready — press Solve')))
      .catch((e) => alive && setStatus(`failed to load model: ${e}`));
    return () => {
      alive = false;
    };
  }, []);

  const valid = cluesValid(puzzle);
  const blanks = puzzle.filter((v) => v === 0).length;
  const editable = !running && frame === null;

  // drop any in-flight solve and return the board to its editable input state
  const toEdit = useCallback(() => {
    runId.current++;
    setRunning(false);
    setFrame(null);
  }, []);

  const setCell = useCallback((i: number, v: number) => {
    setPuzzle((p) => {
      const q = [...p];
      q[i] = v;
      return q;
    });
  }, []);

  const loadPool = () => {
    if (!puzzles.length) return;
    let i = Math.floor(Math.random() * puzzles.length);
    if (puzzles.length > 1) {
      const cur = puzzle.join('');
      while (puzzles[i].puzzle === cur) i = Math.floor(Math.random() * puzzles.length);
    }
    setPuzzle(puzzles[i].puzzle.split('').map(Number));
    setSelected(null);
    toEdit();
    setStatus(ready ? 'ready — press Solve' : status);
  };

  const clearBoard = () => {
    setPuzzle(new Array(N).fill(0));
    setSelected(null);
    toEdit();
    setStatus('enter a puzzle, then press Solve');
  };

  const loadPaste = () => {
    const parsed = parsePuzzle(pasteText);
    if (!parsed) {
      setStatus('paste needs exactly 81 cells (digits 1–9, with 0 or . for blanks)');
      return;
    }
    setPuzzle(parsed);
    setSelected(null);
    toEdit();
    setStatus(ready ? 'loaded — press Solve' : status);
  };

  const onCellClick = (i: number) => {
    if (running) return;
    if (frame !== null) setFrame(null); // re-enter edit mode after a solve
    setSelected(i);
    boardRef.current?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!editable || selected === null) return;
    const k = e.key;
    if (k >= '1' && k <= '9') {
      setCell(selected, k.charCodeAt(0) - 48);
    } else if (k === '0' || k === '.' || k === 'Backspace' || k === 'Delete') {
      setCell(selected, 0);
    } else if (k === 'ArrowLeft') {
      setSelected(Math.max(0, selected - 1));
    } else if (k === 'ArrowRight') {
      setSelected(Math.min(N - 1, selected + 1));
    } else if (k === 'ArrowUp') {
      setSelected(Math.max(0, selected - 9));
    } else if (k === 'ArrowDown') {
      setSelected(Math.min(N - 1, selected + 9));
    } else {
      return;
    }
    e.preventDefault();
  };

  const run = useCallback(async () => {
    if (!ready || running || !valid || blanks === 0) return;
    const id = ++runId.current;
    setRunning(true);
    setSelected(null);
    setStatus('solving…');
    const session = await loadModel();
    const tau = tau1000 / 1000;
    let last: Frame | null = null;
    for await (const f of solve(session, puzzle, { mode, tau, steps })) {
      if (runId.current !== id) return; // superseded
      last = f;
      setFrame(f);
      await sleep(delayRef.current);
    }
    if (runId.current !== id) return;
    setRunning(false);
    if (last) {
      setStatus(
        last.solved
          ? `solved ✓ in ${last.step} step${last.step === 1 ? '' : 's'}`
          : `filled in ${last.step} steps — not a valid grid ✗`,
      );
    }
  }, [ready, running, valid, blanks, puzzle, mode, tau1000, steps]);

  return (
    <div className="wrap">
      <header>
        <h1>SudokuDiT</h1>
        <p>
          A 1.28M-parameter <strong>Diffusion Transformer</strong> solving Sudoku as{' '}
          <strong>masked discrete diffusion</strong> — it fills the cells it is most confident
          about first, MaskGIT-style. Running entirely in your browser via ONNX.
        </p>
        <div className="links">
          <a href="https://github.com/tchauffi/nonet">GitHub</a>
          <a href="https://huggingface.co/tchauffi/sudoku-dit">Model on 🤗 Hub</a>
        </div>
      </header>

      <div className="layout">
        <div className="boardcol">
          <Board
            frame={frame}
            puzzle={puzzle}
            selected={selected}
            editable={editable}
            boardRef={boardRef}
            onCellClick={onCellClick}
            onKeyDown={onKeyDown}
          />
          <div className="hint">
            {editable
              ? 'click a cell, then type 1–9 (0 or ⌫ clears) · or paste a puzzle →'
              : 'editing the grid? click any cell'}
          </div>
        </div>

        <div className="panel">
          <div className="field">
            <label>Decode mode</label>
            <div className="seg">
              <button className={mode === 'confidence' ? 'on' : ''} onClick={() => setMode('confidence')}>
                confidence (adaptive)
              </button>
              <button className={mode === 'iterative' ? 'on' : ''} onClick={() => setMode('iterative')}>
                iterative (fixed)
              </button>
            </div>
          </div>

          {mode === 'confidence' ? (
            <div className="field">
              <label>
                confidence threshold τ = <span className="val">{(tau1000 / 1000).toFixed(3)}</span>
              </label>
              <input
                type="range"
                min={900}
                max={999}
                value={tau1000}
                onChange={(e) => setTau1000(+e.target.value)}
              />
            </div>
          ) : (
            <div className="field">
              <label>
                reveal steps = <span className="val">{steps}</span>
              </label>
              <input type="range" min={1} max={81} value={steps} onChange={(e) => setSteps(+e.target.value)} />
            </div>
          )}

          <div className="field">
            <label>
              animation delay = <span className="val">{delay} ms</span>
            </label>
            <input type="range" min={0} max={400} step={20} value={delay} onChange={(e) => setDelay(+e.target.value)} />
          </div>

          <div className="row">
            <button className="btn primary" onClick={run} disabled={!ready || running || !valid || blanks === 0}>
              {running ? 'solving…' : 'Solve'}
            </button>
          </div>
          <div className="row">
            <button className="btn" onClick={loadPool} disabled={running}>
              New puzzle
            </button>
            <button className="btn" onClick={clearBoard} disabled={running}>
              Clear
            </button>
          </div>

          <div className="field paste-field">
            <label>or paste your own (81 cells)</label>
            <div className="row">
              <input
                className="paste"
                value={pasteText}
                placeholder="53..7....6..195...."
                onChange={(e) => setPasteText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && loadPaste()}
                disabled={running}
              />
              <button className="btn load" onClick={loadPaste} disabled={running}>
                Load
              </button>
            </div>
          </div>

          <div className="status">
            {!valid ? (
              <span className="bad">invalid grid — a digit repeats in a row, column, or box</span>
            ) : (
              <>
                {status}
                {editable && <span> · {blanks} blanks</span>}
              </>
            )}
          </div>

          <div className="legend">
            model confidence (step 0)
            <div className="bar" />
            <div className="ends">
              <span>unsure</span>
              <span>sure</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Board({
  frame,
  puzzle,
  selected,
  editable,
  boardRef,
  onCellClick,
  onKeyDown,
}: {
  frame: Frame | null;
  puzzle: number[];
  selected: number | null;
  editable: boolean;
  boardRef: React.RefObject<HTMLDivElement>;
  onCellClick: (i: number) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
}) {
  const showSolve = frame !== null;
  const grid = showSolve ? frame!.grid : puzzle;
  const clue = showSolve ? frame!.clue : puzzle.map((v) => v !== 0);
  const conf = showSolve ? frame!.conf : null;

  return (
    <div
      className={`board${editable ? ' editing' : ''}`}
      ref={boardRef}
      tabIndex={editable ? 0 : -1}
      onKeyDown={onKeyDown}
    >
      {grid.map((v, i) => {
        const col = i % 9;
        const row = Math.floor(i / 9);
        const cls = ['cell'];
        if (col === 2 || col === 5) cls.push('br');
        if (row === 2 || row === 5) cls.push('bb');
        let style: React.CSSProperties | undefined;

        if (showSolve) {
          if (clue[i]) cls.push('clue');
          else if (v === 0) cls.push('blank');
          else style = { background: tint(conf![i]), color: '#e6edf3' };
        } else {
          cls.push('editable');
          if (i === selected) cls.push('sel');
          if (v !== 0) cls.push('clue');
          else cls.push('blank');
        }

        return (
          <div
            key={i}
            className={cls.join(' ')}
            style={style}
            onClick={() => onCellClick(i)}
          >
            {v === 0 ? '' : v}
          </div>
        );
      })}
    </div>
  );
}
