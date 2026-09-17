"""Hybrid optimiser for Erdős #670 in dimension d: SA (sa670.anneal) -> SLSQP ordering polish (opt670b.polish),
with random starts and starts seeded from optimal 1D Golomb rulers (embedded on a line, perturbed).
Usage: python hybrid670.py d nmin nmax [restarts]"""
import sys, time, json
import numpy as np
from sa670 import anneal
from opt670b import polish, feasible_value, pair_index

GOLOMB = {5: [0,1,4,9,11], 6: [0,1,4,10,12,17], 7: [0,1,4,10,18,23,25], 8: [0,1,4,9,15,22,32,34], 9: [0,1,5,12,25,27,35,41,44],
          10: [0,1,6,10,23,26,34,41,53,55], 11: [0,1,4,13,28,33,47,54,64,70,72], 12: [0,2,6,24,29,40,43,55,68,75,76,85],
          13: [0,2,5,25,37,43,59,70,85,89,98,99,106], 14: [0,4,6,20,35,52,59,77,78,86,89,99,122,127]}

def run(n, d, restarts, rng):
    iu = pair_index(n); N = n * (n - 1) // 2; best = (np.inf, None)
    for r in range(restarts):
        if r % 3 == 0 and n in GOLOMB and d >= 1:
            X = np.zeros((n, d)); X[:, 0] = GOLOMB[n]; X += rng.normal(0, 0.5 + r * 0.1, X.shape)
        elif best[1] is not None and r % 3 == 1:
            X = best[1].reshape(n, d) + rng.normal(0, 0.05 * N, (n, d))
        else:
            X = rng.uniform(-1, 1, (n, d)) * N
        val, X = anneal(X.copy(), n, d, 1_500_000, 0.3 * N, 1e-3, r + 1000 * n)
        try:
            x, D = polish(X.ravel(), n, d, iu, rounds=40); val2 = feasible_value(x, n, d, iu)
        except Exception:
            x, val2 = X.ravel(), val
        if val2 < val: val, X = val2, x.reshape(n, d)
        if val < best[0]: best = (val, X.ravel())
    return best

if __name__ == '__main__':
    d = int(sys.argv[1]); nmin, nmax = int(sys.argv[2]), int(sys.argv[3]); restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    gol = {n: v[-1] for n, v in GOLOMB.items()}
    rng = np.random.default_rng(7); out = {}
    for n in range(nmin, nmax + 1):
        t = time.time(); val, x = run(n, d, restarts, rng); N = n * (n - 1) // 2
        print(f"d={d} n={n}: best diameter = {val:.3f}   C(n,2) = {N}   ratio = {val/N:.3f}   1D optimum = {gol.get(n)} (ratio {gol.get(n, 0)/N:.3f})  [{time.time()-t:.0f}s]", flush=True)
        out[n] = dict(value=float(val), x=x.tolist()); json.dump(out, open(f'hybrid_d{d}.json', 'w'))
