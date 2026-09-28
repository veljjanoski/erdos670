# Written on 27 Sep 2026 by an AI review agent (Claude Opus 5.5, Anthropic) as satellite2.py. Copied here with two changes:
# the default repo path is this folder, and one no-op line (an exec of an empty string) was removed.
# Usage: python satellite2.py m k R seed   (m-mark optimal Golomb ruler + k satellites, R random starts)
"""Optimal Golomb ruler (m marks) + k random satellites, polished by the repo's SLSQP ordering step; report best D/g."""
import sys, os, time, json, numpy as np
sys.path.insert(0, os.environ.get('ERDOS670_REPO', os.path.dirname(os.path.abspath(__file__))))
from opt670b import polish, feasible_value, pair_index
GOL = {9: [0,1,5,12,25,27,35,41,44], 10: [0,1,6,10,23,26,34,41,53,55], 11: [0,1,4,13,28,33,47,54,64,70,72], 12: [0,2,6,24,29,40,43,55,68,75,76,85],
       13: [0,2,5,25,37,43,59,70,85,89,98,99,106], 14: [0,4,6,20,35,52,59,77,78,86,89,99,122,127]}
GN = {10: 55, 11: 72, 12: 85, 13: 106, 14: 127, 15: 151, 16: 177}
m, k, R, seed = map(int, sys.argv[1:5]); A = np.array(GOL[m], float); L = A[-1]; n = m + k; iu = pair_index(n); rng = np.random.default_rng(seed)
best = (np.inf, None); vals = []; t = time.time()
for r in range(R):
    X = np.zeros((n, 2)); X[:m, 0] = A + rng.normal(0, 0.3, m); X[:m, 1] = rng.normal(0, 0.3, m)
    X[m:, 0] = rng.uniform(-0.3 * L, 1.3 * L, k); X[m:, 1] = rng.uniform(1, 1.0 * L, k) * rng.choice([-1, 1], k)
    try:
        x, D = polish(X.ravel(), n, 2, iu, rounds=60); v = feasible_value(x, n, 2, iu)
    except Exception: continue
    vals.append(v)
    if v < best[0]: best = (v, x)
print(f"ruler m={m} + {k} satellites (n={n}): best {best[0]:.3f}, median {np.median(vals):.1f}, #runs {len(vals)};  Golomb G(n)={GN.get(n)}  [{time.time()-t:.0f}s]", flush=True)
json.dump(dict(value=best[0], x=best[1].tolist()), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sat2_{m}_{k}_{seed}.json'), 'w'))
