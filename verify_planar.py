"""Independent check of configs.json for Erdős #670 (needs only mpmath).
For each n: integer coordinates P (units of 1e-4).  Computes all pairwise distances with 60-digit precision,
g = smallest gap between consecutive sorted distances, D = largest distance.  The configuration P/g has all
pairwise distances at least 1 apart and diameter D/g."""
import json
from mpmath import mp, mpf, sqrt
mp.dps = 60
cfg = json.load(open('configs.json'))
for n in sorted(cfg, key=int):
    P = cfg[n]['points']; k = int(n); dists = []
    for i in range(k):
        for j in range(i + 1, k):
            dists.append(sqrt(mpf((P[i][0]-P[j][0])**2 + (P[i][1]-P[j][1])**2)))
    dists.sort(); g = min(dists[t+1] - dists[t] for t in range(len(dists)-1)); D = dists[-1]
    assert len(dists) == k*(k-1)//2 and g > 0
    print(f"n={k}: points={P}")
    print(f"      min gap g = {mp.nstr(g, 12)}, diameter D = {mp.nstr(D, 12)}, certified diameter D/g = {mp.nstr(D/g, 10)}, C(n,2) = {k*(k-1)//2}")
