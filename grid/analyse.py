"""Print the structure of a planar configuration: offsets from the best-fit line, positions along it, and the sorted
distances labelled r (line-line) or S (involving a point far from the line). Usage: python analyse.py FILE.json [far_thr]"""
import sys, json, numpy as np
J = json.load(open(sys.argv[1])); J = J if 'x' in J else {'x': next(iter(J.values()))['points']}; x = np.array(J['x'], float).reshape(-1, 2); n = len(x); thr = float(sys.argv[2]) if len(sys.argv) > 2 else 8
iu = np.triu_indices(n, 1); d = np.linalg.norm(x[iu[0]] - x[iu[1]], axis=1); g = np.diff(np.sort(d)).min(); d /= g; x = x / g
c = x - x.mean(0); _, _, vt = np.linalg.svd(c); u = c @ vt[0]; w = c @ vt[1]
far = set(np.where(np.abs(w - np.median(w)) > thr)[0])
print(f"n={n}  D/g={d.max():.3f}  far points {sorted(far)}")
for i in np.argsort(u): print(f"  pt {i:2d}: along {u[i]-u.min():7.2f}  offset {w[i]-np.median(w):7.2f}{'  FAR' if i in far else ''}")
lab = ['S' if (i in far or j in far) else 'r' for i, j in zip(*iu)]
o = np.argsort(d); print(' '.join(f'{d[k]:.2f}{lab[k]}' for k in o))
