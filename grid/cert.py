"""Certify a found configuration the same way as the repository: scale to minimum gap 1, translate to nonnegative
coordinates, round to integers in units of 1e-4 (as certify.py), then (a) 60-digit mpmath check (certify.verify_config)
and (b) the integer-only upper bound of certify_exact.certify.  Writes config_n{n}_{name}.json in this folder.
Usage: python cert.py FILE.json|FILE.jsonl:i name"""
import sys, os, json, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from certify import verify_config
from certify_exact import certify
src, name = sys.argv[1], sys.argv[2]
if '.jsonl:' in src:
    f, i = src.rsplit(':', 1); rec = json.loads(open(f).read().splitlines()[int(i)])
else:
    rec = json.load(open(src))
X = np.array(rec['x'], float).reshape(-1, 2); n = len(X); iu = np.triu_indices(n, 1)
X = X / np.diff(np.sort(np.linalg.norm(X[iu[0]] - X[iu[1]], axis=1))).min(); X -= X.min(0)
P = [[int(round(c * 10000)) for c in row] for row in X]
g, D, _ = verify_config(P); N, g_low, D_up, bound = certify(P)
up = math.ceil(bound * 10**6) / 10**6                     # integer bound rounded UP to 6 decimals
print(f"n={n}: float value {rec['value']:.5f}; rounded: mpmath D/g = {float(D / g):.8f}; integer bound D/g <= {up:.6f} "
      f"({N} distinct squared distances)")
json.dump({str(n): dict(points=P, unit='1e-4', diameter=round(float(D / g), 8), exact_upper_bound=up)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'config_n{n}_{name}.json'), 'w'), indent=1)
