# Erdős problem #670 in the plane — explicit small configurations

**Problem** (https://www.erdosproblems.com/670). Let A be a set of n points in R^d such that all pairwise distances
differ by at least 1. Is the diameter of A at least (1 + o(1)) n²? Trivially the diameter is at least C(n, 2).
Erdős proved the statement for d = 1 (it is the Erdős–Turán bound for Sidon sets). Ho (2026) disproved it when
d is allowed to grow with n; the fixed-dimension question, in particular d = 2, is open.

**Content of this repository.** Explicit planar configurations, found by a GPU annealing search and then
certified with exact integer coordinates and 60-digit arithmetic, whose distances are pairwise at least 1 apart
and whose diameter is below the length of the optimal Golomb ruler with the same number of marks (the integer
version of the problem on a line). These are upper bounds for small n only; they say nothing about asymptotics.

| n | certified planar diameter | C(n,2) | optimal Golomb ruler (line, integer case) |
|---|---|---|---|
| 5 | 10.1863 | 10 | 11 |
| 6 | 15.3158 | 15 | 17 |
| 7 | 21.5964 | 21 | 25 |
| 8 | 30.9350 | 28 | 34 |
| 9 | 40.9511 | 36 | 44 |
| 10 | 54.8667 | 45 | 55 |
| 11 | 77.1695 | 55 | 72 |
| 12 | 107.3715 | 66 | 85 |

For n = 11, 12 the search did not reach the line values, i.e. it is far from optimal there; those rows are kept
only for completeness. Our line searches with the same code reproduced the Golomb values 17, 25, 34, 44 for
n = 6..9 and were above the optimum for n ≥ 10, so the planar values for n ≥ 10 are certainly not optimal either.

**Certification.** `configs.json` holds, for each n, integer coordinates (units of 10⁻⁴). `verify_planar.py`
(needs only `mpmath`) recomputes all C(n,2) distances to 60 digits, the minimum gap g between consecutive sorted
distances and the diameter D; the configuration scaled by 1/g has all distances ≥ 1 apart and diameter D/g, which
is the value in the table.

**Search.** `gpu670b.py` (CuPy + numba): one simulated-annealing chain per GPU thread on the scale-invariant
objective diameter / (soft-minimum gap), with an incremental sorted distance list; the best chains are polished on
the CPU by a constrained (SLSQP) step and a least-squares fit to a stretched distance pattern (`opt670b.py`).
`certify.py` rounds and certifies. Earlier, weaker optimisers: `opt670.py`, `sa670.py`, `hybrid670.py`.
`interleave.py` is an unrelated side experiment.

**Observations (not results).** The best small configurations are genuinely two-dimensional (not close to a
line), with sorted distances forming an almost consecutive sequence with a few missing values.
