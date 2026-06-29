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

// a brighter variant of the same blue->green ramp, for text/bars on a dark bg
function tintBright(p: number): string {
  const n = Math.min(Math.max((p - 0.6) / 0.4, 0), 1);
  return `hsl(${222 - 72 * n} 70% 64%)`;
}

// a plausible 9-way softmax peaked at digit d with mass `conf` — for the schema's
// "estimate" bars (the real per-cell distribution comes from the model).
function distFor(d: number, conf: number): number[] {
  const p = new Array(9).fill(0);
  p[d - 1] = conf;
  const rest = 1 - conf;
  p[d % 9] += rest * 0.6; // a near-miss digit
  p[(d + 3) % 9] += rest * 0.4; // and another
  return p;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ---- "how it works" schematic: a tiny 3x3 illustration of the decode loop ----
// (not a real Sudoku unit — just shows predict-confidence -> reveal-surest -> repeat)
const SCHEMA_GIVEN: Record<number, number> = { 0: 5, 2: 9, 6: 4 };
// masked cells in the order the model reveals them (most confident first)
const SCHEMA_STEPS: { i: number; d: number; conf: number }[] = [
  { i: 4, d: 7, conf: 0.99 },
  { i: 7, d: 2, conf: 0.96 },
  { i: 1, d: 8, conf: 0.92 },
  { i: 5, d: 1, conf: 0.86 },
  { i: 3, d: 6, conf: 0.79 },
  { i: 8, d: 3, conf: 0.71 },
];

const rowOf = (i: number) => Math.floor(i / 9);
const colOf = (i: number) => i % 9;
const boxOf = (i: number) => Math.floor(rowOf(i) / 3) * 3 + Math.floor(colOf(i) / 3);

// the row, column, and box indices for unit u (0..8)
function unit(u: number): { row: number[]; col: number[]; box: number[] } {
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
  return { row, col, box };
}

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

// Per-cell flag: true if this filled cell repeats a digit within its row,
// column, or box. Powers the live red conflict highlight.
function findConflicts(grid: number[]): boolean[] {
  const bad = new Array(N).fill(false);
  const mark = (idxs: number[]) => {
    const at: Record<number, number[]> = {};
    for (const i of idxs) {
      const v = grid[i];
      if (v === 0) continue;
      (at[v] ||= []).push(i);
    }
    for (const v in at) if (at[v].length > 1) for (const i of at[v]) bad[i] = true;
  };
  for (let u = 0; u < 9; u++) {
    const { row, col, box } = unit(u);
    mark(row);
    mark(col);
    mark(box);
  }
  return bad;
}

// how many of each digit 1..9 are already placed (to fade out finished digits)
function digitCounts(grid: number[]): number[] {
  const c = new Array(10).fill(0);
  for (const v of grid) c[v]++;
  return c;
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

  const conflicts = findConflicts(puzzle);
  const valid = !conflicts.some(Boolean);
  const blanks = puzzle.filter((v) => v === 0).length;
  const editable = !running && frame === null;
  const counts = digitCounts(puzzle);

  // drop any in-flight solve and return the board to its editable input state
  const toEdit = useCallback(() => {
    runId.current++;
    setRunning(false);
    setFrame(null);
  }, []);

  const setCell = useCallback((i: number, v: number) => {
    setPuzzle((p) => {
      if (p[i] === v) return p;
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
    setSelected(0);
    toEdit();
    setStatus('enter a puzzle, then press Solve');
    boardRef.current?.focus();
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

  const enterEdit = (i: number) => {
    if (frame !== null) setFrame(null); // re-enter edit mode after a solve
    setSelected(i);
    boardRef.current?.focus();
  };

  const onCellClick = (i: number) => {
    if (running) return;
    enterEdit(i);
  };

  // tap a number-pad key to fill the selected cell (no keyboard needed)
  const padInput = (v: number) => {
    if (!editable || selected === null) return;
    setCell(selected, v);
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

      <HowItWorks />

      <div className="layout">
        <div className="boardcol">
          <Board
            frame={frame}
            puzzle={puzzle}
            selected={selected}
            conflicts={conflicts}
            editable={editable}
            boardRef={boardRef}
            onCellClick={onCellClick}
            onKeyDown={onKeyDown}
          />

          {editable && (
            <>
              <NumberPad
                counts={counts}
                disabled={selected === null}
                onDigit={padInput}
                onErase={() => padInput(0)}
              />
              <div className="hint">
                {selected === null
                  ? 'tap a cell to select it, then tap a number — or use your keyboard / paste'
                  : 'tap a number to fill · ⌫ clears · arrow keys move · or paste a board →'}
              </div>
            </>
          )}
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
            <label>or paste a board (81 cells)</label>
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
              <span className="bad">conflict — the cells in red repeat a digit in a row, column, or box</span>
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
  conflicts,
  editable,
  boardRef,
  onCellClick,
  onKeyDown,
}: {
  frame: Frame | null;
  puzzle: number[];
  selected: number | null;
  conflicts: boolean[];
  editable: boolean;
  boardRef: React.RefObject<HTMLDivElement>;
  onCellClick: (i: number) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
}) {
  const showSolve = frame !== null;
  const grid = showSolve ? frame!.grid : puzzle;
  const clue = showSolve ? frame!.clue : puzzle.map((v) => v !== 0);
  const conf = showSolve ? frame!.conf : null;
  const selVal = selected !== null ? grid[selected] : 0;

  // is cell i a peer (same row/col/box) of the selected cell?
  const isPeer = (i: number) =>
    selected !== null &&
    i !== selected &&
    (rowOf(i) === rowOf(selected) || colOf(i) === colOf(selected) || boxOf(i) === boxOf(selected));

  return (
    <div
      className={`board${editable ? ' editing' : ''}`}
      ref={boardRef}
      tabIndex={editable ? 0 : -1}
      onKeyDown={onKeyDown}
    >
      {grid.map((v, i) => {
        const cls = ['cell'];
        if (colOf(i) === 2 || colOf(i) === 5) cls.push('br');
        if (rowOf(i) === 2 || rowOf(i) === 5) cls.push('bb');
        const style: React.CSSProperties = {};

        if (showSolve) {
          if (clue[i]) cls.push('clue');
          else if (v === 0) cls.push('blank');
          else {
            style.background = tint(conf![i]);
            style.color = '#e6edf3';
          }
        } else {
          cls.push('editable');
          const sel = i === selected;
          const same = !sel && v !== 0 && v === selVal;
          if (sel) cls.push('sel');
          // backgrounds: selected > same-digit > peer > given clue > blank
          if (sel) style.background = '#1f2937';
          else if (same) style.background = '#243349';
          else if (isPeer(i)) style.background = '#161d2b';
          else if (v !== 0) style.background = 'var(--clue)';
          // text: conflicts in red, otherwise normal entered-digit color
          if (v !== 0) style.color = conflicts[i] ? '#f85149' : 'var(--text)';
        }

        return (
          <div key={i} className={cls.join(' ')} style={style} onClick={() => onCellClick(i)}>
            {v === 0 ? '' : v}
          </div>
        );
      })}
    </div>
  );
}

function NumberPad({
  counts,
  disabled,
  onDigit,
  onErase,
}: {
  counts: number[];
  disabled: boolean;
  onDigit: (v: number) => void;
  onErase: () => void;
}) {
  return (
    <div className="pad">
      {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => (
        <button
          key={d}
          className={counts[d] >= 9 ? 'done' : ''}
          disabled={disabled}
          onClick={() => onDigit(d)}
          title={counts[d] >= 9 ? `all nine ${d}s placed` : undefined}
        >
          {d}
        </button>
      ))}
      <button className="erase" disabled={disabled} onClick={onErase} title="clear cell" aria-label="erase">
        ⌫
      </button>
    </div>
  );
}

// Paper-style data-flow figure of a single forward pass:
// input board -> SudokuDiT -> per-cell softmax p(digit) -> argmax prediction.
// Animated by cycling which masked cell is being "queried"; already-decided
// cells stay filled, so it also conveys the most-confident-first reveal order.
function HowItWorks() {
  const L = SCHEMA_STEPS.length;
  const [k, setK] = useState(0);

  useEffect(() => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      setK(L); // show the finished grid, no motion
      return;
    }
    const id = setInterval(() => setK((p) => (p >= L ? 0 : p + 1)), 1500);
    return () => clearInterval(id);
  }, [L]);

  const solved = k >= L;
  const active = Math.min(k, L - 1);
  const step = SCHEMA_STEPS[active];
  const probs = distFor(step.d, step.conf);

  return (
    <section className="schema" aria-label="how the model maps a board to a digit">
      <div className="schema-head">A single forward pass — how the denoiser reads a board and estimates a digit</div>

      <div className="arch">
        {/* input board */}
        <div className="stage">
          <div className="schema-grid" aria-hidden="true">
            {Array.from({ length: 9 }, (_, i) => {
              if (i in SCHEMA_GIVEN) {
                return (
                  <div key={i} className="scell given">
                    {SCHEMA_GIVEN[i]}
                  </div>
                );
              }
              const oi = SCHEMA_STEPS.findIndex((x) => x.i === i);
              const revealed = solved || oi < k;
              const queried = !solved && oi === k;
              const cls = ['scell', revealed ? 'filled' : 'masked'];
              if (queried) cls.push('cand');
              return (
                <div
                  key={i}
                  className={cls.join(' ')}
                  style={{ background: tint(SCHEMA_STEPS[oi].conf), opacity: revealed || queried ? 1 : 0.4 }}
                >
                  {revealed ? SCHEMA_STEPS[oi].d : '?'}
                </div>
              );
            })}
          </div>
          <span className="stage-sub">
            input
            <br />
            81 tokens · 0=[mask]
          </span>
        </div>

        <Arrow label="embed + t" />

        {/* model */}
        <div className="stage">
          <div className="dit" aria-hidden="true">
            <span className="dit-name">SudokuDiT</span>
            {[0, 1, 2, 3].map((n) => (
              <div key={n} className="dit-layer" />
            ))}
          </div>
          <span className="stage-sub">
            token+pos+box
            <br />
            ×4 DiT · d=128
          </span>
        </div>

        <Arrow label="softmax" />

        {/* per-cell estimate */}
        <div className="stage">
          <div className="estimate" aria-hidden="true">
            <div className="bars">
              {probs.map((p, j) => (
                <div key={j} className="barcol">
                  <div className="bartrack">
                    <div
                      className="barfill"
                      style={{
                        height: `${Math.max(4, Math.round(p * 100))}%`,
                        background: j + 1 === step.d ? tintBright(step.conf) : 'var(--line)',
                      }}
                    />
                  </div>
                  <span className={`barlab${j + 1 === step.d ? ' on' : ''}`}>{j + 1}</span>
                </div>
              ))}
            </div>
            <div className="pred">
              <span className="pred-k">argmax →</span>
              <span className="pred-d" style={{ color: tintBright(step.conf) }}>
                {step.d}
              </span>
              <span className="pred-c">{Math.round(step.conf * 100)}%</span>
            </div>
          </div>
          <span className="stage-sub">
            estimate · p(digit)
            <br />
            per cell, 9-way
          </span>
        </div>
      </div>

      <p className="schema-phase">
        {solved ? (
          <>
            <span className="k">complete ✓</span> — re-reads the board and repeats
          </>
        ) : (
          <>
            reading the whole board, the denoiser estimates <span className="k">p(digit)</span> for every empty
            cell, then fills its <span className="k">argmax</span> — surest first, MaskGIT-style.
          </>
        )}
      </p>
    </section>
  );
}

function Arrow({ label }: { label: string }) {
  return (
    <div className="arrow" aria-hidden="true">
      <span className="arrow-line">→</span>
      <span className="arrow-lab">{label}</span>
    </div>
  );
}
