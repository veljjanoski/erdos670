"""Grid-start search for #670 (planar): a fixed base configuration plus ONE new point, started from every position of
a grid and polished with the repo's opt670b.polish (SLSQP with the distance order fixed). Deterministic (no randomness).
Usage: python gridpolish.py BASE lam step padx pady half w W tag [thr]
  BASE: 'ruler:0,1,6,...' (marks on the x-axis, multiplied by lam), or FILE.json ({"x": [...]}), or FILE.jsonl:i (line i).
  Grid: step apart over the base's bounding box enlarged by padx*span (x) and pady*span (y), span = largest side;
  half = 1 keeps only y >= step (use for a ruler base: y -> -y is a symmetry).  This process takes the grid points
  with index = w mod W.  Every polished value < thr (default inf) is appended to gp_{tag}_{w}.jsonl.
One BLAS thread per process, below-normal priority."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_v] = '1'   # before numpy import
import sys, time, json, numpy as np
if os.name == 'nt':
    import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)  # BELOW_NORMAL (-1 = this process)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.environ.get('ERDOS670_REPO', _here), os.path.dirname(_here)]
from opt670b import polish, feasible_value, pair_index

base, lam, step, padx, pady, half = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
w, W, tag = int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]; thr = float(sys.argv[10]) if len(sys.argv) > 10 else np.inf
if base.startswith('ruler:'):
    a = np.array([float(v) for v in base[6:].split(',')]) * lam; B = np.column_stack([a, np.zeros_like(a)])
else:
    if '.jsonl:' in base:
        f, i = base.rsplit(':', 1); B = np.array(json.loads(open(f).read().splitlines()[int(i)])['x'], float).reshape(-1, 2)
    else:
        B = np.array(json.load(open(base))['x'], float).reshape(-1, 2)
    iu0 = pair_index(len(B)); B = B / np.diff(np.sort(np.linalg.norm(B[iu0[0]] - B[iu0[1]], axis=1))).min()   # min gap 1
m = len(B); n = m + 1; iu = pair_index(n)
lo, hi = B.min(0), B.max(0); span = (hi - lo).max()
xs = np.arange(lo[0] - padx * span, hi[0] + padx * span + 1e-9, step)
ys = np.arange(step, hi[1] + pady * span + 1e-9, step) if half else np.arange(lo[1] - pady * span, hi[1] + pady * span + 1e-9, step)
grid = [(x, y) for x in xs for y in ys]; mine = grid[w::W]
OUT = os.path.join(_here, f'gp_{tag}_{w}.jsonl')
t = time.time(); best = np.inf; nb = 0
for c, (px, py) in enumerate(mine):
    X = np.vstack([B, [px, py]])
    try:
        x, _ = polish(X.ravel(), n, 2, iu, rounds=60); v = feasible_value(x, n, 2, iu)
    except Exception:
        continue
    if v < thr:
        nb += 1
        with open(OUT, 'a') as f: f.write(json.dumps(dict(value=v, x=x.tolist(), start=[px, py], lam=lam, base=base)) + '\n')
    best = min(best, v)
    if (c + 1) % 500 == 0:
        print(f"  [{tag} w{w}] {c+1}/{len(mine)} starts, best {best:.3f}, #<thr {nb}  [{time.time()-t:.0f}s]", flush=True)
print(f"[{tag} w{w}] done {len(mine)} of {len(grid)} grid starts: best {best:.3f}, #<thr {nb}  [{time.time()-t:.0f}s]", flush=True)
