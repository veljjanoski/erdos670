"""GPU search for Erdős #670 in dimension d (1 or 2): n points, minimise F = diam / min_gap over the sorted pairwise
distances (scale-invariant; X/min_gap is feasible with diameter F).  One simulated-annealing chain per GPU thread.
State per thread: coordinates X[n][d], distance array dist[N] (pair-indexed), sorted array S[N] with pair ids P[N].
Move: perturb one point; update its n-1 distances; re-insert them into the sorted list (O(n N)).
Host loop: launch T chains, collect the best, polish on CPU (SLSQP ordering step from opt670b), re-seed around the elite.
Usage: python gpu670.py d nmin nmax [--threads T] [--iters I] [--rounds R]
"""
import sys, time, json, argparse
import numpy as np, cupy as cp
from opt670b import polish, feasible_value, pair_index

MAXN, MAXP = 20, 190          # n <= 20 -> N <= 190

KSRC = r'''
#define MAXN 20
#define MAXP 190
__device__ float softobj(const float* S, int N) {
    // diam / softmin(gaps), softmin = (sum g^-6)^(-1/6); g clipped at 1e-3
    float acc = 0.0f;
    for (int a = 0; a + 1 < N; a++) { float g = S[a+1] - S[a]; if (g < 1e-3f) g = 1e-3f; float r = 1.0f / g; float r2 = r * r; acc += r2 * r2 * r2; }
    float sm = powf(acc, -1.0f / 6.0f);
    return S[N-1] / sm;
}
extern "C" __global__
void anneal(float* Xall, int n, int d, int iters, float T0, float T1, unsigned long long seed, float* bestF, float* bestX) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n * (n - 1) / 2;
    float X[MAXN * 2]; float dist[MAXP]; float S[MAXP]; int P[MAXP];
    int pi[MAXP], pj[MAXP];
    for (int i = 0; i < n * d; i++) X[i] = Xall[(size_t)t * n * d + i];
    int m = 0;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) { pi[m] = i; pj[m] = j; m++; }
    // rng: xorshift64*
    unsigned long long s = seed ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)(t + 1));
    #define RND() (s ^= s << 13, s ^= s >> 7, s ^= s << 17, (float)((s * 0x2545F4914F6CDD1DULL) >> 40) * (1.0f / 16777216.0f))
    for (int p = 0; p < N; p++) {
        float acc = 0; for (int k = 0; k < d; k++) { float df = X[pi[p]*d+k] - X[pj[p]*d+k]; acc += df * df; }
        dist[p] = sqrtf(acc);
    }
    // initial sort (insertion)
    for (int p = 0; p < N; p++) { S[p] = dist[p]; P[p] = p; }
    for (int a = 1; a < N; a++) { float v = S[a]; int q = P[a]; int b = a - 1; while (b >= 0 && S[b] > v) { S[b+1] = S[b]; P[b+1] = P[b]; b--; } S[b+1] = v; P[b+1] = q; }
    float cur = softobj(S, N);
    float best = cur; float bX[MAXN * 2]; for (int i = 0; i < n * d; i++) bX[i] = X[i];
    float scale = 1.0f; int acc = 0;
    float oldc[2]; float olddist[MAXN];
    for (int it = 0; it < iters; it++) {
        float T = T0 * powf(T1 / T0, (float)it / (float)iters);
        int i = (int)(RND() * n); if (i >= n) i = n - 1;
        for (int k = 0; k < d; k++) { oldc[k] = X[i*d+k]; X[i*d+k] += (2.0f * RND() - 1.0f) * scale; }
        // update distances of pairs involving i
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            int p = (i < j) ? (i * (2 * n - i - 1) / 2 + (j - i - 1)) : (j * (2 * n - j - 1) / 2 + (i - j - 1));
            olddist[j] = dist[p];
            float a2 = 0; for (int k = 0; k < d; k++) { float df = X[i*d+k] - X[j*d+k]; a2 += df * df; }
            float v = sqrtf(a2); dist[p] = v;
            // move entry p in the sorted list to its new place
            int pos = 0; while (P[pos] != p) pos++;
            // shift up
            while (pos + 1 < N && S[pos+1] < v) { S[pos] = S[pos+1]; P[pos] = P[pos+1]; pos++; }
            while (pos > 0 && S[pos-1] > v) { S[pos] = S[pos-1]; P[pos] = P[pos-1]; pos--; }
            S[pos] = v; P[pos] = p;
        }
        float nw = softobj(S, N);
        if (nw <= cur || RND() < expf(-(nw - cur) / T)) {
            cur = nw; acc++;
            if (cur < best) { best = cur; for (int q = 0; q < n * d; q++) bX[q] = X[q]; }
        } else {
            // revert: coordinates and distances (re-insert old values)
            for (int k = 0; k < d; k++) X[i*d+k] = oldc[k];
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                int p = (i < j) ? (i * (2 * n - i - 1) / 2 + (j - i - 1)) : (j * (2 * n - j - 1) / 2 + (i - j - 1));
                float v = olddist[j]; dist[p] = v;
                int pos = 0; while (P[pos] != p) pos++;
                while (pos + 1 < N && S[pos+1] < v) { S[pos] = S[pos+1]; P[pos] = P[pos+1]; pos++; }
                while (pos > 0 && S[pos-1] > v) { S[pos] = S[pos-1]; P[pos] = P[pos-1]; pos--; }
                S[pos] = v; P[pos] = p;
            }
        }
        if ((it & 1023) == 1023) { float rate = acc / 1024.0f; acc = 0; if (rate > 0.3f) scale *= 1.2f; else if (rate < 0.15f) scale *= 0.8f; }
    }
    bestF[t] = best;
    for (int q = 0; q < n * d; q++) bestX[(size_t)t * n * d + q] = bX[q];
}
'''
_k = cp.RawKernel(KSRC, 'anneal')

GOLOMB = {5: [0,1,4,9,11], 6: [0,1,4,10,12,17], 7: [0,1,4,10,18,23,25], 8: [0,1,4,9,15,22,32,34], 9: [0,1,5,12,25,27,35,41,44],
          10: [0,1,6,10,23,26,34,41,53,55], 11: [0,1,4,13,28,33,47,54,64,70,72], 12: [0,2,6,24,29,40,43,55,68,75,76,85],
          13: [0,2,5,25,37,43,59,70,85,89,98,99,106], 14: [0,4,6,20,35,52,59,77,78,86,89,99,122,127],
          15: [0,4,20,30,57,59,62,76,100,111,123,136,144,145,151], 16: [0,1,4,11,26,32,56,68,76,115,117,134,150,163,168,177]}

def gpu_round(X0, n, d, iters, T0, T1, seed):
    T = X0.shape[0]
    Xg = cp.asarray(X0.reshape(T, -1).astype(np.float32)); bF = cp.zeros(T, dtype=cp.float32); bX = cp.zeros_like(Xg)
    _k(((T + 127) // 128,), (128,), (Xg, np.int32(n), np.int32(d), np.int32(iters), np.float32(T0), np.float32(T1), np.uint64(seed), bF, bX))
    cp.cuda.Device().synchronize()
    return bF.get().astype(np.float64), bX.get().astype(np.float64).reshape(T, n, d)

from scipy.optimize import least_squares
def pattern_snap(x, n, d, iu):
    """stretch gaps < 1 to 1 -> target sorted pattern; least-squares fit of sorted distances to it; return coordinates"""
    X = x.reshape(n, d); dd = np.sqrt(((X[iu[0]] - X[iu[1]])**2).sum(1)); order = np.argsort(dd); s = dd[order]
    gaps = np.maximum(np.diff(s), 1.0); target = np.concatenate([[max(s[0], 1.0)], max(s[0], 1.0) + np.cumsum(gaps)])
    t_by_pair = np.empty_like(dd); t_by_pair[order] = target
    def resid(z):
        Y = z.reshape(n, d); return np.sqrt(((Y[iu[0]] - Y[iu[1]])**2).sum(1)) - t_by_pair
    r = least_squares(resid, x.copy(), method='lm', max_nfev=2000)
    return r.x

def search(n, d, threads, iters, rounds, rng, verbose=True, seed_golomb=True):
    N = n * (n - 1) // 2; iu = pair_index(n)
    X0 = rng.uniform(-1, 1, (threads, n, d)) * N
    if seed_golomb and n in GOLOMB:                 # seed a few chains from the optimal ruler
        for t in range(min(64, threads)):
            X0[t] = 0; X0[t, :, 0] = GOLOMB[n]; X0[t] += rng.normal(0, 0.3 + 0.05 * t, (n, d))
    best = (np.inf, None); elite = None
    for r in range(rounds):
        T0 = 0.3 * N * (0.5 ** r); T1 = 1e-3
        F, X = gpu_round(X0, n, d, iters, T0, T1, seed=1000 * n + r)
        order = np.argsort(F); elite = X[order[:32]]
        # polish the top chains on CPU
        for e in range(8):
            try:
                x0 = elite[e].ravel(); F[order[e]] = feasible_value(x0, n, d, iu)      # true (hard) objective of the elite
                for x in (x0, pattern_snap(x0, n, d, iu)):
                    x, D = polish(x, n, d, iu, rounds=40); v = feasible_value(x, n, d, iu)
                    if v < F[order[e]]: elite[e] = x.reshape(n, d); F[order[e]] = v
            except Exception: pass
        if F[order[0]] < best[0]: best = (float(F[order[0]]), elite[0].copy())
        if verbose: print(f"    n={n} round {r}: best {best[0]:.4f}  (round best {F.min():.4f}, median {np.median(F):.2f})", flush=True)
        # re-seed: half random, half around elite
        X0 = rng.uniform(-1, 1, (threads, n, d)) * N
        for t in range(threads // 2):
            X0[t] = elite[t % len(elite)] * (1 + rng.normal(0, 0.02)) + rng.normal(0, 0.02 * N, (n, d))
    return best

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('d', type=int); ap.add_argument('nmin', type=int); ap.add_argument('nmax', type=int)
    ap.add_argument('--threads', type=int, default=8192); ap.add_argument('--iters', type=int, default=200000); ap.add_argument('--rounds', type=int, default=6); ap.add_argument('--noseed', action='store_true')
    a = ap.parse_args(); rng = np.random.default_rng(3); out = {}
    for n in range(a.nmin, a.nmax + 1):
        t = time.time(); val, X = search(n, a.d, a.threads, a.iters, a.rounds, rng, seed_golomb=not a.noseed); N = n * (n - 1) // 2
        gol = GOLOMB[n][-1] if n in GOLOMB else None
        print(f"d={a.d} n={n}: best diameter = {val:.4f}   C(n,2) = {N}   delta = {val/N - 1:.4f}   1D optimum = {gol} (delta {gol/N - 1 if gol else float('nan'):.4f})  [{time.time()-t:.0f}s]", flush=True)
        out[n] = dict(value=val, X=X.tolist()); json.dump(out, open(f'gpu_d{a.d}.json', 'w'))
