"""Erdős #670, fixed dimension d: n points in R^d, all pairwise distances differ by >= 1; minimise the diameter.
Scale-invariant objective F(X) = diam(X) / min_gap(X), where min_gap = min over consecutive sorted distances.
Any configuration X gives the feasible configuration X / min_gap with diameter F(X).  Random restarts + Nelder-Mead/Powell.
Usage: python opt670.py d nmin nmax [restarts]"""
import sys, time, json
import numpy as np
from scipy.optimize import minimize

def dists(X):
    n = len(X); iu = np.triu_indices(n, 1)
    return np.sqrt(((X[iu[0]] - X[iu[1]])**2).sum(1))

def F(x, n, d):
    X = x.reshape(n, d); s = np.sort(dists(X)); gaps = np.diff(s)
    g = gaps.min()
    if g <= 1e-12: return 1e9
    return s[-1] / g

def run(n, d, restarts, rng, seed_x=None):
    best = (np.inf, None)
    for r in range(restarts):
        if seed_x is not None and r % 2 == 0:
            x0 = seed_x + rng.normal(0, 0.05 * seed_x.std(), seed_x.shape)
        else:
            x0 = rng.uniform(-1, 1, n * d) * n
        res = minimize(F, x0, args=(n, d), method='Nelder-Mead', options=dict(maxiter=4000 * n, xatol=1e-9, fatol=1e-10))
        res = minimize(F, res.x, args=(n, d), method='Powell', options=dict(maxiter=2000 * n, xtol=1e-10, ftol=1e-12))
        if res.fun < best[0]: best = (res.fun, res.x)
    return best

if __name__ == '__main__':
    d = int(sys.argv[1]); nmin, nmax = int(sys.argv[2]), int(sys.argv[3]); restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    golomb = {2: 1, 3: 3, 4: 6, 5: 11, 6: 17, 7: 25, 8: 34, 9: 44, 10: 55, 11: 72, 12: 85, 13: 106, 14: 127, 15: 151, 16: 177}
    rng = np.random.default_rng(0); out = {}
    for n in range(nmin, nmax + 1):
        t = time.time(); val, x = run(n, d, restarts, rng)
        N = n * (n - 1) // 2
        print(f"d={d} n={n}: best diam/min_gap = {val:.3f}   trivial C(n,2) = {N}   ratio = {val/N:.3f}   1D Golomb = {golomb.get(n)}  [{time.time()-t:.0f}s]", flush=True)
        out[n] = dict(value=float(val), x=x.tolist())
        json.dump(out, open(f'best_d{d}.json', 'w'))
