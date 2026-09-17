"""Sub-question for the two-line construction: two sets A, B of k reals each; the union of the two positive
difference sets must be 1-separated.  Minimise L / min_gap where L = max(diam A, diam B).  SA.  Compare with k^2
(both sets Erdős–Turán-tight and perfectly interleaved) and with the optimal Golomb ruler of 2k marks."""
import sys, time
import numpy as np
from numba import njit

@njit(cache=True)
def obj(x, k, buf):
    m = 0
    for s in range(2):
        for i in range(k):
            for j in range(i + 1, k):
                buf[m] = abs(x[s*k + i] - x[s*k + j]); m += 1
    b = np.sort(buf[:m]); g = 1e300
    if b[0] < g: g = b[0]
    for t in range(m - 1):
        if b[t+1] - b[t] < g: g = b[t+1] - b[t]
    if g <= 1e-12: return 1e300
    L = max(x[:k].max() - x[:k].min(), x[k:].max() - x[k:].min())
    return L / g

@njit(cache=True)
def anneal(x, k, iters, T0, T1, seed):
    np.random.seed(seed); buf = np.zeros(k * (k - 1))
    cur = obj(x, k, buf); best = cur; bx = x.copy(); scale = 1.0; acc = 0
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters); i = np.random.randint(2 * k); old = x[i]
        x[i] += np.random.normal() * scale; new = obj(x, k, buf)
        if new <= cur or np.random.random() < np.exp(-(new - cur) / T):
            cur = new; acc += 1
            if cur < best: best = cur; bx = x.copy()
        else: x[i] = old
        if it % 1000 == 999:
            r = acc / 1000.0; acc = 0
            if r > 0.3: scale *= 1.2
            elif r < 0.15: scale /= 1.2
    return best, bx

golomb = {6: 17, 8: 34, 10: 55, 12: 85, 14: 127, 16: 177}
for k in range(3, 9):
    t = time.time(); best = (1e300, None)
    for r in range(12):
        x = np.random.default_rng(r).uniform(0, k * k, 2 * k)
        v, bx = anneal(x, k, 3_000_000, k * k / 2, 1e-3, r)
        if v < best[0]: best = (v, bx)
    print(f"k={k}: L/gap = {best[0]:.2f}   k^2 = {k*k}   2*C(k,2) = {k*(k-1)}   Golomb(2k) = {golomb.get(2*k)}   A = {np.round(np.sort(best[1][:k]),2)}  B = {np.round(np.sort(best[1][k:]),2)}  [{time.time()-t:.0f}s]", flush=True)
