# Erdős problem #670 in the plane — explicit small configurations

**Problem** (https://www.erdosproblems.com/670). Let A be a set of n points in R^d such that all pairwise distances
differ by at least 1. Is the diameter of A at least (1 + o(1)) n²? Trivially the diameter is at least C(n, 2).
Erdős proved the statement for d = 1 (it is the Erdős–Turán bound for Sidon sets). Ho (2026) disproved it when
d is allowed to grow with n; the fixed-dimension question, in particular d = 2, is open.

**Content of this repository.** Explicit planar configurations, found by a GPU annealing search (for n = 11 and 12
by a different construction, see below) and then certified with exact integer coordinates and 60-digit arithmetic,
whose distances are pairwise at least 1 apart and whose diameter, for n = 5..12, is below the length of the optimal
Golomb ruler with the same number of marks (the integer version of the problem on a line). We do not know whether
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
| 11 | 59.1476 (see below) | 55 | 72 |
| 12 | 81.1377 (see below) | 66 | 85 |

For n = 11 and 12 the GPU search only reached 77.1695 and 107.3715 (these sets are still in `configs.json`); the
table values for n = 11 and 12 come from the ruler-plus-points constructions below. Our line searches with the same
code reproduced the Golomb values 17, 25, 34, 44 for n = 6..9 and stayed above the Golomb values for n = 10..12, so
the GPU search is weak from n = 10 on and the planar value for n = 10 is probably not optimal (not proved). ArneD
(forum of the problem page, 17 Sep 2026) independently re-derived essentially the same n = 5 configuration, with
diameter about 10.18506; that is also our unrounded value (`hybrid_d2.json`), and the table value 10.1863 is
slightly higher only because the coordinates are rounded to 10⁻⁴ for certification.

**n = 11, first construction (added 28 Sep 2026; improved below).** Start from the optimal 10-mark Golomb ruler 0,
1, 6, 10, 23, 26, 34, 41, 53, 55, slightly perturbed, add one random point off the line and polish with
`opt670b.polish`: the best of 120 random starts (`satellite2.py 10 1 120 1`, `sat2_10_1.log`) gives a floating-point
optimum with D/g = 65.7251 (`sat2_10_1_1.json`); scaled to minimum gap 1, translated and rounded to integers in
units of 10⁻⁴ (as in `certify.py`), it is the set in `config_n11_satellite.json`, with certified D/g = 65.73153 (≤
65.7317), below the Golomb value 72. Ten of its points lie close to a slightly bent line and the eleventh far from
it. This construction and set were found on 27 Sep 2026 by an AI review agent (Claude Opus 5.5, Anthropic) running
on this repository; the certificate was re-checked here with 60-digit arithmetic (`verify_planar.py
config_n11_satellite.json`) and with the integer-only bound of `certify_exact.py config_n11_satellite.json`. We have
not checked the literature for better n = 11 sets.

**n = 11 and 12, grid starts (added 29 Sep 2026).** The construction of the first n = 11 set above (an optimal
Golomb ruler plus one point), with deterministic grid starts instead of random starts. `grid/gridpolish.py` uses the
same base configuration for every start, places one new point at each point of a square grid of spacing 1 around the
base, and polishes all points with the SLSQP step of `opt670b.py`; `grid/gp_run.sh` runs it in parallel. There is no
randomness. (i) n = 11: the base is the optimal 10-mark Golomb ruler 0, 1, 6, 10, 23, 26, 34, 41, 53, 55 on a line,
stretched by the factor 1.02 (the best of the factors 1.00, 1.02, ..., 1.10 tried; `bash gp_run.sh 12
ruler:0,1,6,10,23,26,34,41,53,55 1.02 1 0.4 1.0 1 n11 70`); the best result has D/g = 60.0830
(`grid/n11_start_float.json`). Removing one point of that set and re-inserting it by the same kind of grid search
(`bash reinsert.sh n11_start_float.json r11`; point indices from 0) gives D/g = 59.1433 (`grid/n11_best_float.json`,
point 5 removed); rounded as in `certify.py`, it is the set in `config_n11_grid.json`, with certified D/g = 59.14756
(≤ 59.1489 by `certify_exact.py`). (ii) n = 12: twelve n = 11 sets from the same re-insertion run were tried as
bases; the best result comes from the one with D/g = 61.4471 (`grid/n12_base_float.json`, point 6 removed): adding
one point by the grid search (`bash gp_run.sh 12 n12_base_float.json 1 1 0.4 0.4 0 n12 95`) gives D/g = 81.1311
(`grid/n12_best_float.json`), and the rounded set in `config_n12_grid.json` has certified D/g = 81.13761 (≤
81.1377), below the Golomb value 85. Both sets are ten points close to a line (within about 3% of its length) plus
one (n = 11) or two (n = 12, on the same side) points far from it. Other bases gave larger values: the two optimal
11-mark Golomb rulers plus one point reached only 86.61 and 85.44, the best n = 11 set (59.14) plus one point 82.36,
and removing and re-inserting single points of the two final sets gave no further improvement. For n = 5..10, the
optimal (n − 1)-mark Golomb rulers plus one point did not improve the table values. The certificates were re-checked
with `verify_planar.py`, `certify_exact.py` and `referee_check.py` (a separately written checker with exact integer
interval bounds for every distance). These are local optima of a heuristic search, not proved optimal; we have not
checked the literature for better sets.

**Certification.** `configs.json` holds, for each n, integer coordinates (units of 10⁻⁴) of the GPU-search sets; its
n = 11 and 12 entries are the older GPU-search sets (77.1695 and 107.3715); the n = 11 and 12 sets of the table are
in `config_n11_grid.json` and `config_n12_grid.json`, and the first n = 11 construction (65.7315) is in
`config_n11_satellite.json`. `verify_planar.py [file]` (needs only `mpmath`, default file `configs.json`) recomputes
all C(n,2) distances to 60 digits, the minimum gap g between consecutive sorted distances and the diameter D; the
configuration scaled by 1/g has all distances ≥ 1 apart and diameter D/g, which for the sets of the table is the
table value (rounded up to 4 decimals). `certify_exact.py file` gives a rigorous upper bound for D/g using integers
only (no floating point), within 0.0013 of each table value; for the sets of the table (n = 5..12) every bound is
below the Golomb value. `referee_check.py file` is a separately written check of the same numbers (exact integer
interval bounds for every distance).

**Search.** `gpu670b.py` (CuPy + numba): one simulated-annealing chain per GPU thread on the scale-invariant
objective diameter / (soft-minimum gap), with an incremental sorted distance list; the best chains are polished on
the CPU by a constrained (SLSQP) step and a least-squares fit to a stretched distance pattern (`opt670b.py`).
`certify.py` rounds and certifies. Earlier, weaker optimisers: `opt670.py`, `sa670.py`, `hybrid670.py`.
`interleave.py` is an unrelated side experiment. `grid/` holds the grid-start search of 29 Sep 2026 (see above).

**Observations (not results).** For n = 5..10 the best configurations found are genuinely two-dimensional (not close
to a line), with sorted distances forming an almost consecutive sequence with a few missing values; the n = 11 and
12 sets above are instead close to a line plus one or two points.
