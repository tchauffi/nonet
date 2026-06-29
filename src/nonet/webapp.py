
"""Tiny zero-dependency web UI to watch the SudokuDiT solver fill a board.

Run::

    python -m nonet.webapp                       # newest checkpoint, port 8000
    python -m nonet.webapp --ckpt path/to.pt --port 8123

Then open http://localhost:8000 . The page asks the backend for a random
validation puzzle, posts it to ``/api/solve``, and animates the reveal trace
(``SudokuSolver.solve(..., return_trace=True)``) cell by cell.

Only the Python standard library is used for serving -- the frontend is the
single HTML page below.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from nonet.model import SudokuDiT
from nonet.pipeline import MASK_TOKEN, SudokuSolver
from nonet.schedueler import LinearScheduler
from nonet.sudokuer import SudokuJudge

# Populated in main(): the loaded solver, judge, device, and a pool of puzzles.
STATE: dict = {}


def newest_checkpoint() -> str | None:
    cks = glob.glob("checkpoints/**/*.pt", recursive=True)
    return max(cks, key=os.path.getmtime) if cks else None


def load_solver(ckpt: str, num_heads: int, device: str) -> SudokuSolver:
    """Rebuild a SudokuDiT from a checkpoint, inferring its size from the weights."""
    sd = torch.load(ckpt, map_location=device)
    hidden = sd["square_embed.weight"].shape[1]
    grid = sd["square_embed.weight"].shape[0] - 1                 # vocab = digits + MASK
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    mlp_ratio = sd["blocks.0.mlp.0.weight"].shape[0] / hidden
    model = SudokuDiT(hidden_size=hidden, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, num_blocks=n_blocks, grid_size=grid).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {ckpt}  (hidden={hidden}, blocks={n_blocks}, heads={num_heads}, grid={grid})")
    return SudokuSolver(model, LinearScheduler())


def build_pool(n: int = 200, min_blank: int = 0, max_blank: int = 81, sample: int = 8000):
    """Validation puzzles spread evenly across the full difficulty range.

    The dataset is ~54% mid-difficulty (40-49 blanks) and failures live in the rare
    60+ blank tail, so a plain random pool almost never shows a failure. We bucket by
    blank count and round-robin across buckets, giving an even easy->impossible spread
    so the demo reliably surfaces the boards where the model breaks. Narrow
    ``min_blank``/``max_blank`` to focus on one difficulty band instead.
    """
    from datasets import load_dataset
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="validation")
    buckets: dict[int, list] = {}
    for idx in random.sample(range(len(ds)), min(sample, len(ds))):
        row = ds[idx]
        puzzle = [int(c) for c in row["puzzle"]]
        b = puzzle.count(0)
        if min_blank <= b <= max_blank:
            buckets.setdefault(b // 8, []).append(
                {"puzzle": puzzle, "solution": [int(c) for c in row["solution"]]})

    order = sorted(buckets)
    pool, i = [], 0
    while len(pool) < n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]:
            pool.append(buckets[k].pop())
        i += 1
    random.shuffle(pool)
    blanks = [sum(1 for v in p["puzzle"] if v == 0) for p in pool]
    span = f"{min(blanks)}-{max(blanks)} blanks" if pool else "empty"
    print(f"puzzle pool: {len(pool)} boards, even spread across {span}")
    return pool


def solve_trace(puzzle: list[int], steps: int = 27, tau: float | None = None,
                tiebreak: bool = False):
    solver: SudokuSolver = STATE["solver"]
    device = STATE["device"]
    x = torch.tensor(puzzle, dtype=torch.long, device=device).unsqueeze(0)
    if tau is not None:
        # confidence mode: fill every cell the model is >= tau sure of each step;
        # num_steps is just a loop cap (81 >= max blanks, so it never has to dump).
        _, trace, init_conf = solver.solve(x, num_steps=81, return_trace=True,
                                           constraint_tiebreak=tiebreak, conf_threshold=tau)
    else:
        # iterative mode: a fixed number of reveal steps.
        _, trace, init_conf = solver.solve(x, num_steps=steps, return_trace=True,
                                           constraint_tiebreak=tiebreak)
    final = trace[-1]
    solved = bool(STATE["judge"].is_solved(final.to(device))[0].item())
    return {
        "clues": [c != MASK_TOKEN for c in puzzle],
        "trace": [t[0].tolist() for t in trace],
        "conf": [round(c, 4) for c in init_conf[0].tolist()],  # step-0 per-cell certainty
        "solved": solved,
        "steps": len(trace),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/puzzle":
            pool = STATE["pool"]
            board = random.choice(pool) if pool else {"puzzle": [0] * 81, "solution": [0] * 81}
            self._send(200, json.dumps(board))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/solve":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        puzzle = req.get("puzzle", [0] * 81)
        steps = int(req.get("steps", 27))
        tau = req.get("tau", None)
        tau = float(tau) if tau is not None else None
        tiebreak = bool(req.get("tiebreak", False))
        try:
            self._send(200, json.dumps(solve_trace(puzzle, steps, tau, tiebreak)))
        except Exception as exc:  # surface model errors to the page
            self._send(500, json.dumps({"error": str(exc)}))


def main():
    p = argparse.ArgumentParser(description="Web UI for the SudokuDiT solver.")
    p.add_argument("--ckpt", default=None, help="checkpoint .pt (default: newest under checkpoints/)")
    p.add_argument("--num-heads", type=int, default=4, help="must match training (not stored in ckpt)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--min-blank", type=int, default=0, help="drop puzzles with fewer blanks")
    p.add_argument("--max-blank", type=int, default=81,
                   help="drop puzzles with more blanks (default keeps all, incl. failures)")
    args = p.parse_args()

    ckpt = args.ckpt or newest_checkpoint()
    if not ckpt:
        raise SystemExit("no checkpoint found; pass --ckpt or train first")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    STATE["device"] = device
    STATE["solver"] = load_solver(ckpt, args.num_heads, device)
    STATE["judge"] = SudokuJudge()
    STATE["pool"] = build_pool(n=args.pool_size, min_blank=args.min_blank, max_blank=args.max_blank)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  ▸ open  http://{args.host}:{args.port}  (Ctrl-C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SudokuDiT · live solve</title>
<style>
  :root{
    --bg:#11131a; --panel:#181b24; --line:#2a2f3d; --line-bold:#5b6478;
    --ink:#e8eaf0; --muted:#8b92a6; --clue:#f4f6fb; --fill:#6ea8fe;
    --good:#52d39a; --bad:#ef6a6a; --accent:#8b7bff;
  }
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:flex;flex-direction:column;align-items:center;
    gap:22px;padding:40px 16px;background:
      radial-gradient(900px 500px at 50% -10%, #1c2030 0%, var(--bg) 60%);
    color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
  h1{margin:0;font-size:22px;font-weight:650;letter-spacing:.2px}
  h1 span{color:var(--accent)}
  .sub{color:var(--muted);margin-top:-14px;font-size:13px}
  .board{display:grid;grid-template-columns:repeat(9,1fr);
    width:min(92vw,468px);aspect-ratio:1;background:var(--line-bold);
    border:3px solid var(--line-bold);border-radius:12px;overflow:hidden;
    box-shadow:0 18px 50px rgba(0,0,0,.45)}
  .cell{position:relative;display:flex;align-items:center;justify-content:center;background:var(--panel);
    font-size:clamp(16px,4.6vw,26px);font-weight:600;font-variant-numeric:tabular-nums;
    color:var(--fill);user-select:none;transition:background .3s,color .25s}
  .cell.clue{color:var(--clue);font-weight:700;background:#1f2330}
  .cell.empty .d{color:transparent}
  .cell.pop{animation:pop .45s ease}
  .cell.bad{outline:2px solid var(--bad);outline-offset:-2px;border-radius:3px}
  .cell .conf{position:absolute;right:3px;bottom:1px;font-size:9px;font-weight:600;
    line-height:1;color:rgba(255,255,255,.62);font-variant-numeric:tabular-nums}
  .cell.clue .conf,.cell.empty .conf{display:none}
  .grad{display:inline-block;width:84px;height:11px;border-radius:3px;vertical-align:-1px;
    background:linear-gradient(90deg,hsl(222,50%,26%),hsl(186,54%,29%),hsl(150,58%,32%))}
  /* thicker rules between 3x3 boxes */
  .cell:nth-child(9n+1){border-left:0}
  .r3{box-shadow:inset 0 2px 0 var(--line-bold)}
  .c3{box-shadow:inset 2px 0 0 var(--line-bold)}
  .c3.r3{box-shadow:inset 2px 0 0 var(--line-bold),inset 0 2px 0 var(--line-bold)}
  @keyframes pop{0%{transform:scale(.4);opacity:0}60%{transform:scale(1.18)}100%{transform:scale(1)}}
  .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:center}
  button{appearance:none;border:1px solid var(--line);background:#222636;color:var(--ink);
    padding:10px 18px;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;
    transition:transform .08s,background .2s,border-color .2s}
  button:hover{background:#2a2f44;border-color:var(--line-bold)}
  button:active{transform:translateY(1px)}
  button.primary{background:var(--accent);border-color:var(--accent);color:#0c0d14}
  button:disabled{opacity:.45;cursor:not-allowed}
  .speed{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}
  select{background:#222636;color:var(--ink);border:1px solid var(--line);border-radius:8px;
    padding:5px 8px;font-size:13px;font-weight:600;cursor:pointer}
  .status{min-height:24px;font-size:14px;color:var(--muted);display:flex;gap:14px;align-items:center}
  .status b{color:var(--ink);font-weight:600}
  .tag{padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700}
  .tag.good{background:rgba(82,211,154,.16);color:var(--good)}
  .tag.bad{background:rgba(239,106,106,.16);color:var(--bad)}
  .legend{display:flex;gap:18px;color:var(--muted);font-size:12px}
  .legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
</style>
</head>
<body>
  <h1>Sudoku<span>DiT</span> — watch it solve</h1>
  <p class="sub">masked-diffusion solver, revealing the most confident cells first</p>
  <div class="board" id="board"></div>
  <div class="controls">
    <button id="new">New puzzle</button>
    <button id="solve" class="primary">Solve ▸</button>
    <label class="speed" title="confidence: fill every cell >= τ sure (adaptive steps). iterative: a fixed number of reveal steps.">
      mode
      <select id="mode">
        <option value="confidence">confidence (adaptive)</option>
        <option value="iterative">iterative (fixed)</option>
      </select>
    </label>
    <label class="speed" id="tauctl" title="adaptive reveal: each step fills every cell the model is at least τ confident of (lower τ = bolder, fewer steps)">
      τ <input id="tau" type="range" min="900" max="999" value="999" step="1">
      <b id="tauval">0.999</b>
    </label>
    <label class="speed" id="stepsctl" style="display:none" title="fixed number of reveal steps: more = better on hard puzzles, but slower">
      steps <input id="steps" type="range" min="1" max="81" value="27" step="1">
      <b id="stepsval">27</b>
    </label>
    <label class="speed">speed
      <input id="speed" type="range" min="40" max="600" value="220" step="20">
    </label>
    <label class="speed" title="among equally-confident cells, fill the most constrained (e.g. naked singles) first">
      <input id="tiebreak" type="checkbox"> obvious cells first
    </label>
  </div>
  <div class="status" id="status"></div>
  <div class="legend">
    <span><i style="background:#1f2330"></i>clue</span>
    <span>model fill <i class="grad"></i> low&nbsp;→&nbsp;high confidence</span>
    <span><i style="outline:2px solid var(--bad);outline-offset:-2px;background:transparent"></i>wrong vs solution</span>
  </div>
<script>
const boardEl=document.getElementById('board');
const statusEl=document.getElementById('status');
const speedEl=document.getElementById('speed');
const tauEl=document.getElementById('tau');
const stepsEl=document.getElementById('steps');
const modeEl=document.getElementById('mode');
const fmtTau=()=>(tauEl.value/1000).toFixed(3);
tauEl.oninput=()=>{document.getElementById('tauval').textContent=fmtTau();};
stepsEl.oninput=()=>{document.getElementById('stepsval').textContent=stepsEl.value;};
function syncMode(){
  const adaptive=modeEl.value==='confidence';
  document.getElementById('tauctl').style.display=adaptive?'':'none';
  document.getElementById('stepsctl').style.display=adaptive?'none':'';
}
modeEl.onchange=syncMode; syncMode();
const cells=[];
for(let i=0;i<81;i++){
  const d=document.createElement('div');
  d.className='cell empty';
  const r=Math.floor(i/9), c=i%9;
  if(r%3===0 && r!==0) d.classList.add('r3');
  if(c%3===0 && c!==0) d.classList.add('c3');
  d.innerHTML='<span class="d"></span><span class="conf"></span>';
  boardEl.appendChild(d); cells.push(d);
}
let cur={puzzle:[],solution:[],clues:[]};
let animating=false;

// conf in [0,1] -> a cool blue(unsure) .. green(sure) tint.
// red is reserved for "wrong vs solution", so low confidence never reads as wrong.
function tint(p){
  const n=Math.max(0,Math.min(1,(p-0.6)/0.4));
  return `hsl(${Math.round(222-72*n)},${Math.round(50+8*n)}%,${Math.round(26+6*n)}%)`;
}
function setCell(cell,val,{clue=false,conf=null,wrong=false}={}){
  const dEl=cell.querySelector('.d'), cEl=cell.querySelector('.conf');
  cell.classList.remove('clue','empty','bad','pop');
  cell.style.background=''; cell.style.color=''; cEl.textContent='';
  if(val===0){cell.classList.add('empty'); dEl.textContent='0'; return;}
  dEl.textContent=val;
  if(clue){cell.classList.add('clue'); return;}
  if(conf!=null){
    cell.style.background=tint(conf);
    cell.style.color='#eef1f8';
    cEl.textContent = conf>=0.9995 ? '100' : String(Math.round(conf*100));
  }
  if(wrong) cell.classList.add('bad');
}

function render(grid,clues){
  for(let i=0;i<81;i++) setCell(cells[i],grid[i],{clue:!!(clues && clues[i])});
}

async function newPuzzle(){
  const r=await fetch('/api/puzzle'); cur=await r.json();
  cur.clues=cur.puzzle.map(v=>v!==0);
  render(cur.puzzle,cur.clues);
  const blanks=cur.puzzle.filter(v=>v===0).length;
  statusEl.innerHTML=`<span><b>${blanks}</b> blanks to fill</span>`;
}

function sleep(ms){return new Promise(r=>setTimeout(r,ms));}

async function solve(){
  if(animating) return;
  animating=true; toggle(true);
  statusEl.innerHTML='<span>thinking…</span>';
  const tiebreak=document.getElementById('tiebreak').checked;
  const adaptive=modeEl.value==='confidence';
  const body=adaptive ? {puzzle:cur.puzzle,tau:tauEl.value/1000,tiebreak}
                      : {puzzle:cur.puzzle,steps:parseInt(stepsEl.value),tiebreak};
  const r=await fetch('/api/solve',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  const res=await r.json();
  if(res.error){statusEl.innerHTML=`<span class="tag bad">error: ${res.error}</span>`;animating=false;toggle(false);return;}
  const {trace,clues,conf}=res; cur.clues=clues;
  let prev=trace[0].slice();
  render(prev,clues);
  for(let s=1;s<trace.length;s++){
    const g=trace[s];
    for(let i=0;i<81;i++){
      if(g[i]!==prev[i]){
        const wrong=cur.solution && cur.solution[i] && g[i]!==cur.solution[i];
        setCell(cells[i],g[i],{conf:conf[i],wrong});
        const cell=cells[i];
        cell.classList.remove('pop'); void cell.offsetWidth; cell.classList.add('pop');
      }
    }
    prev=g.slice();
    statusEl.innerHTML=`<span>step <b>${s}</b>/${trace.length-1}</span>`;
    await sleep(parseInt(speedEl.value));
  }
  const tag=res.solved?'<span class="tag good">✓ valid solution</span>'
                       :'<span class="tag bad">✗ has conflicts</span>';
  const how=adaptive?`τ=${fmtTau()}`:'fixed';
  statusEl.innerHTML=`<span>done in <b>${trace.length-1}</b> steps (${how})</span> ${tag}`;
  animating=false; toggle(false);
}

function toggle(on){document.getElementById('new').disabled=on;document.getElementById('solve').disabled=on;}
document.getElementById('new').onclick=()=>{if(!animating)newPuzzle();};
document.getElementById('solve').onclick=solve;
newPuzzle();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
