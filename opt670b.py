"""Erdős #670 in fixed dimension d: minimise the diameter of n points in R^d with all pairwise distances >= 1 apart.
Alternating scheme: given the current ordering pi of the N pairs by distance, solve the smooth NLP
   minimise D  s.t.  dist_{pi(k+1)} - dist_{pi(k)} >= 1 (k < N),  dist_{pi(N)} <= D
with SLSQP (variables: coordinates and D); re-sort; repeat until the ordering is stable.  Many random restarts.
Usage: python opt670b.py d nmin nmax [restarts]"""
import sys, time, json
import numpy as np
from scipy.optimize import minimize

def pair_index(n):
    return np.triu_indices(n, 1)

def dvec(X, iu):
    return np.sqrt(((X[iu[0]] - X[iu[1]])**2).sum(1))

def solve_order(x0, D0, n, d, order, iu):
    """order: array of pair indices sorted by increasing distance."""
    N = len(order)
    def obj(z): return z[-1]
    def obj_grad(z): g = np.zeros_like(z); g[-1] = 1; return g
    def cons(z):
        X = z[:-1].reshape(n, d); dd = dvec(X, iu)[order]
        return np.concatenate([dd[1:] - dd[:-1] - 1.0, [z[-1] - dd[-1]], [dd[0] - 1.0]])
    def cons_jac(z):
        X = z[:-1].reshape(n, d); diff = X[iu[0]] - X[iu[1]]; dd = np.sqrt((diff**2).sum(1)); dd[dd == 0] = 1e-12
        # gradient of dist_m wrt coordinates: d/dX_i = diff/dd, d/dX_j = -diff/dd
        G = np.zeros((len(dd), n * d))
        for m in range(len(dd)):
            i, j = iu[0][m], iu[1][m]; g = diff[m] / dd[m]
            G[m, i*d:(i+1)*d] = g; G[m, j*d:(j+1)*d] = -g
        Go = G[order]
        J = np.zeros((N + 1, n * d + 1))
        J[:N-1, :-1] = Go[1:] - Go[:-1]
        J[N-1, :-1] = -Go[-1]; J[N-1, -1] = 1
        J[N, :-1] = Go[0]
        return J
    z0 = np.concatenate([x0, [D0]])
    res = minimize(obj, z0, jac=obj_grad, constraints=[dict(type='ineq', fun=cons, jac=cons_jac)], method='SLSQP',
                   options=dict(maxiter=500, ftol=1e-12))
    return res.x[:-1], res.x[-1], res.success

def polish(x, n, d, iu, rounds=30):
    X = x.reshape(n, d); dd = dvec(X, iu); s = np.sort(dd); g = np.diff(s).min()
    x = x / max(g, 1e-9); X = x.reshape(n, d); dd = dvec(X, iu); D = dd.max()
    for r in range(rounds):
        order = np.argsort(dd)
        x_new, D_new, ok = solve_order(x, D, n, d, order, iu)
        dd_new = dvec(x_new.reshape(n, d), iu); s = np.sort(dd_new); g = np.diff(s).min()
        if g < 1 - 1e-7:                       # infeasible: rescale to feasibility
            x_new = x_new / g; dd_new = dd_new / g; D_new = dd_new.max()
        if D_new < D - 1e-9:
            x, dd, D = x_new, dd_new, D_new
        else:
            break
    return x, D

def feasible_value(x, n, d, iu):
    dd = dvec(x.reshape(n, d), iu); s = np.sort(dd); g = np.diff(s).min()
    return dd.max() / g

def run(n, d, restarts, rng):
    iu = pair_index(n); best = (np.inf, None); N = n * (n - 1) // 2
    for r in range(restarts):
        if best[1] is not None and r % 3 == 1:
            x0 = best[1] * (1 + rng.normal(0, 0.15)) + rng.normal(0, 0.3 * np.sqrt(N), n * d)
        else:
            x0 = rng.uniform(-1, 1, n * d) * N
        try:
            x, D = polish(x0, n, d, iu)
        except Exception:
            continue
        val = feasible_value(x, n, d, iu)
        if val < best[0]: best = (val, x)
    return best

if __name__ == '__main__':
    d = int(sys.argv[1]); nmin, nmax = int(sys.argv[2]), int(sys.argv[3]); restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    golomb = {2: 1, 3: 3, 4: 6, 5: 11, 6: 17, 7: 25, 8: 34, 9: 44, 10: 55, 11: 72, 12: 85, 13: 106, 14: 127, 15: 151, 16: 177, 17: 199, 18: 216, 19: 246, 20: 283}
    rng = np.random.default_rng(1); out = {}
    for n in range(nmin, nmax + 1):
        t = time.time(); val, x = run(n, d, restarts, rng); N = n * (n - 1) // 2
        print(f"d={d} n={n}: best diameter = {val:.3f}   C(n,2) = {N}   ratio = {val/N:.3f}   1D Golomb = {golomb.get(n)}  [{time.time()-t:.0f}s]", flush=True)
        out[n] = dict(value=float(val), x=x.tolist()); json.dump(out, open(f'best_d{d}.json', 'w'))
