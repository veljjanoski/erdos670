"""Simulated annealing for Erdős #670 in fixed dimension d: minimise F(X) = diam(X) / min_gap(X) (scale-invariant;
X / min_gap is a feasible configuration with diameter F).  Moves: perturb one point.  numba-compiled inner loop.
Usage: python sa670.py d nmin nmax [iters] [runs]"""
import sys, time, json
import numpy as np
from numba import njit

@njit(cache=True)
def objective(X, n, d, buf):
    m = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for k in range(d): s += (X[i, k] - X[j, k])**2
            buf[m] = np.sqrt(s); m += 1
    b = np.sort(buf[:m])
    g = 1e300
    for k in range(m - 1):
        if b[k+1] - b[k] < g: g = b[k+1] - b[k]
    if g <= 1e-12: return 1e300
    return b[-1] / g

@njit(cache=True)
def anneal(X, n, d, iters, T0, T1, seed):
    np.random.seed(seed)
    buf = np.zeros(n * (n - 1) // 2)
    cur = objective(X, n, d, buf); best = cur; bestX = X.copy()
    scale = 1.0
    acc = 0
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        i = np.random.randint(n)
        old = X[i].copy()
        for k in range(d): X[i, k] += np.random.normal() * scale
        new = objective(X, n, d, buf)
        if new <= cur or np.random.random() < np.exp(-(new - cur) / T):
            cur = new; acc += 1
            if cur < best: best = cur; bestX = X.copy()
        else:
            X[i] = old
        if it % 1000 == 999:
            rate = acc / 1000.0; acc = 0
            if rate > 0.3: scale *= 1.2
            elif rate < 0.15: scale /= 1.2
    return best, bestX

if __name__ == '__main__':
    d = int(sys.argv[1]); nmin, nmax = int(sys.argv[2]), int(sys.argv[3])
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 2_000_000; runs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    golomb = {5: 11, 6: 17, 7: 25, 8: 34, 9: 44, 10: 55, 11: 72, 12: 85, 13: 106, 14: 127, 15: 151, 16: 177, 17: 199, 18: 216, 19: 246, 20: 283}
    out = {}
    for n in range(nmin, nmax + 1):
        t = time.time(); N = n * (n - 1) // 2; best = (1e300, None)
        for r in range(runs):
            X = np.random.default_rng(r).uniform(-1, 1, (n, d)) * N
            val, bx = anneal(X, n, d, iters, 0.5 * N, 1e-3, r + 100 * n)
            if val < best[0]: best = (val, bx)
        print(f"d={d} n={n}: best diameter = {best[0]:.3f}   C(n,2) = {N}   ratio = {best[0]/N:.3f}   1D Golomb = {golomb.get(n)}  [{time.time()-t:.0f}s]", flush=True)
        out[n] = dict(value=float(best[0]), X=best[1].tolist()); json.dump(out, open(f'sa_d{d}.json', 'w'))
