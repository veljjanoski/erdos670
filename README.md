# Erdős problem #670 in the plane — explicit small configurations

**Problem** (https://www.erdosproblems.com/670). Let A be a set of n points in R^d such that all pairwise distances
differ by at least 1. Is the diameter of A at least (1 + o(1)) n²? Trivially the diameter is at least C(n, 2).
Erdős proved the statement for d = 1 (it is the Erdős–Turán bound for Sidon sets). Ho (2026) disproved it when
d is allowed to grow with n; the fixed-dimension question, in particular d = 2, is open.

**Content of this repository.** Explicit planar configurations, found by a GPU annealing search (for n = 11 by a
different construction, see below) and then certified with exact integer coordinates and 60-digit arithmetic, whose
distances are pairwise at least 1 apart and whose diameter, for n = 5..11, is below the length of the optimal Golomb
ruler with the same number of marks (the integer version of the problem on a line). We do not know whether
real-valued points on a line can do better than the optimal Golomb ruler, so the comparison is with integer rulers
only; at n = 10 the margin is only 0.13. These are upper bounds for small n only; they say nothing about
asymptotics.

| n | certified planar diameter | C(n,2) | optimal Golomb ruler (line, integer case) |
|---|---|---|---|
| 5 | 10.1863 | 10 | 11 |
| 6 | 15.3158 | 15 | 17 |
| 7 | 21.5964 | 21 | 25 |
| 8 | 30.9350 | 28 | 34 |
| 9 | 40.9511 | 36 | 44 |
| 10 | 54.8667 | 45 | 55 |
| 11 | 65.7315 (see below) | 55 | 72 |
| 12 | 107.3715 | 66 | 85 |

For n = 11 the GPU search only reached 77.1695, but a different construction beats the Golomb value (see below). For
n = 12 the search did not reach the Golomb value, i.e. it is far from optimal there; that row is kept only for
completeness. Our line searches with the same code reproduced the Golomb values 17, 25, 34, 44 for n = 6..9 and
stayed above the Golomb values for n = 10..12, so the planar value for n = 10 is probably not optimal either (not
proved). ArneD (forum of the problem page, 17 Sep 2026) independently re-derived essentially the same n = 5
configuration, with diameter about 10.18506; that is also our unrounded value (`hybrid_d2.json`), and the table
value 10.1863 is slightly higher only because the coordinates are rounded to 10⁻⁴ for certification.

**n = 11 (added 28 Sep 2026).** Start from the optimal 10-mark Golomb ruler 0, 1, 6, 10, 23, 26, 34, 41, 53, 55,
slightly perturbed, add one random point off the line and polish with `opt670b.polish`: the best of 120 random
starts (`satellite2.py 10 1 120 1`, `sat2_10_1.log`) gives a floating-point optimum with D/g = 65.7251
(`sat2_10_1_1.json`); scaled to minimum gap 1, translated and rounded to integers in units of 10⁻⁴ (as in
`certify.py`), it is the set in `config_n11_satellite.json`, with certified D/g = 65.73153 (≤ 65.7317), below the
Golomb value 72. Ten of its points lie close to a slightly bent line and the eleventh far from it. This construction
and set were found on 27 Sep 2026 by an AI review agent (Claude Opus 5.5, Anthropic) running on this repository; the
certificate was re-checked here with 60-digit arithmetic (`verify_planar.py config_n11_satellite.json`) and with the
integer-only bound of `certify_exact.py config_n11_satellite.json`. We have not checked the literature for better n
= 11 sets.

**Certification.** `configs.json` holds, for each n, integer coordinates (units of 10⁻⁴) of the GPU-search sets; its
n = 11 entry is the older set (77.1695), and the n = 11 set of the table is in `config_n11_satellite.json`.
`verify_planar.py [file]` (needs only `mpmath`, default file `configs.json`) recomputes all C(n,2) distances to 60
digits, the minimum gap g between consecutive sorted distances and the diameter D; the configuration scaled by 1/g
has all distances ≥ 1 apart and diameter D/g, which is the table value (rounded to 4 decimals). `certify_exact.py
file` gives a rigorous upper bound for D/g using integers only (no floating point), within 0.0005 of each table
value; for n = 5..11 every bound is below the Golomb value.

**Search.** `gpu670b.py` (CuPy + numba): one simulated-annealing chain per GPU thread on the scale-invariant
objective diameter / (soft-minimum gap), with an incremental sorted distance list; the best chains are polished on
the CPU by a constrained (SLSQP) step and a least-squares fit to a stretched distance pattern (`opt670b.py`).
`certify.py` rounds and certifies. Earlier, weaker optimisers: `opt670.py`, `sa670.py`, `hybrid670.py`.
`interleave.py` is an unrelated side experiment.

**Observations (not results).** For n = 5..10 the best configurations found are genuinely two-dimensional (not close
to a line), with sorted distances forming an almost consecutive sequence with a few missing values; the n = 11 set
above is instead close to a line plus one point.