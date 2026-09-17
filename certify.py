"""Certify planar configurations for Erdős #670.
Input: best configurations (JSON from gpu670b.py).  For each n: scale so that the minimum gap between consecutive
sorted distances is 1, round coordinates to integers in units of 1e-4, then verify with 60-digit arithmetic (mpmath):
  g = min gap between consecutive sorted distances,  D = diameter.
The set scaled by 1/g has all pairwise distances >= 1 apart and diameter D/g.  Output: configs.json + a table.
Anyone can re-check with verify_config() below (needs only mpmath)."""
import json, sys
import numpy as np
from mpmath import mp, mpf, sqrt

mp.dps = 60

def verify_config(P):
    """P: list of integer coordinate pairs. Returns (min_gap, diameter, sorted distances) in the given units."""
    n = len(P); dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = P[i][0] - P[j][0]; dy = P[i][1] - P[j][1]
            dists.append(sqrt(mpf(dx * dx + dy * dy)))
    dists.sort()
    g = min(dists[k + 1] - dists[k] for k in range(len(dists) - 1))
    return g, dists[-1], dists

if __name__ == '__main__':
    files = sys.argv[1:]
    out = {}
    for f in files:
        d = json.load(open(f))
        for n, v in d.items():
            n = int(n); X = np.array(v['X'] if 'X' in v else v['x']).reshape(n, 2)
            iu = np.triu_indices(n, 1); dd = np.sqrt(((X[iu[0]] - X[iu[1]])**2).sum(1)); s = np.sort(dd); g0 = np.diff(s).min()
            X = X / g0; X -= X.min(0)                      # min gap 1, nonnegative coordinates
            P = [[int(round(c * 10000)) for c in row] for row in X]
            g, D, dists = verify_config(P)
            ratio = D / g
            if n not in out or float(ratio) < out[n]['diameter']:
                out[n] = dict(points=P, unit='1e-4', min_gap=float(g), diameter_raw=float(D), diameter=float(ratio),
                              N=n * (n - 1) // 2, delta=float(ratio) / (n * (n - 1) // 2) - 1)
    golomb = {5: 11, 6: 17, 7: 25, 8: 34, 9: 44, 10: 55, 11: 72, 12: 85}
    json.dump({str(k): out[k] for k in sorted(out)}, open('configs.json', 'w'), indent=1)
    print("| n | certified diameter (distances >= 1 apart) | C(n,2) | delta | best on a line (optimal Golomb ruler) |")
    print("|---|---|---|---|---|")
    for n in sorted(out):
        print(f"| {n} | {out[n]['diameter']:.4f} | {out[n]['N']} | {out[n]['delta']:.4f} | {golomb.get(n)} |")
