#!/usr/bin/env python3
"""
Independent referee checker for Erdos problem #670 (planar case).

Property being checked: a finite set of points in the plane whose pairwise
Euclidean distances, sorted, differ pairwise (i.e. consecutive gaps in the
sorted list) by at least 1. For an integer-coordinate point set, let

    g = min gap between consecutive values in the sorted list of pairwise
        distances
    D = the largest pairwise distance

Scaling the point set by 1/g makes all sorted distances at least 1 apart,
with diameter D/g. This script computes D/g from the raw integer
coordinates only, by two independent methods, and checks it against a
per-n limit table (optimal Golomb ruler lengths).

This file does NOT import or read any other file in this project tree.
Everything is recomputed from the point coordinates using only the
Python standard library.

Method 1 -- Decimal, high precision (~50+ significant digits):
    Each squared distance is an exact Python integer (sum of squares of
    integer coordinate differences). Decimal(s).sqrt() at working
    precision 60 gives each true distance to 60 significant digits.
    Sorting, taking consecutive differences (g) and the maximum (D) gives
    D/g to about 12+ correct decimal digits after the division.

Method 2 -- rigorous integer interval bound (no floating point, no Decimal):
    For a non-negative integer s and a chosen scale exponent K, note that
    10**(2K) is a perfect square, so sqrt(s * 10**(2K)) = sqrt(s) * 10**K
    exactly. math.isqrt computes the exact floor of the integer square
    root using only integer arithmetic, so

        L(s) = isqrt(s * 10**(2K))                    satisfies
        L(s) <= sqrt(s) * 10**K < L(s) + 1

    Let U(s) = L(s) if L(s)**2 == s*10**(2K) (s*10**(2K) is itself a
    perfect square) else L(s) + 1. Then for every s:

        L(s) / 10**K  <=  sqrt(s)  <=  U(s) / 10**K

    Distances are sorted the same way whether you sort by s or by sqrt(s)
    (sqrt is strictly increasing on non-negative reals), so sorting by the
    exact integer s is itself exact -- no rounding is involved in ordering.
    For consecutive sorted squared distances (s_i, s_{i+1}):

        true gap = sqrt(s_{i+1}) - sqrt(s_i) >= (L(s_{i+1}) - U(s_i)) / 10**K

    so a rigorous lower bound (scaled by 10**K) for g is

        g_lower_scaled = min over consecutive pairs of (L(s_{i+1}) - U(s_i))

    and a rigorous upper bound (scaled by 10**K) for D is

        D_upper_scaled = U(s_max)

    Because both bounds carry the same scale factor 10**K, it cancels
    exactly in the ratio, giving an exact rational (Fraction) upper bound:

        D / g  <=  D_upper_scaled / g_lower_scaled

    This bound is entirely integer arithmetic (via math.isqrt, which is
    exact for arbitrarily large integers) until the very last display
    step, so it is rigorous independent of any floating point or Decimal
    rounding behavior, and independent of Method 1.

Usage:
    python referee_check.py FILE.json [FILE2.json ...]
"""

import sys
import json
import math
from decimal import Decimal, getcontext
from fractions import Fraction

# Optimal Golomb-ruler lengths, used as the pass/fail limit per n.
LIMIT_TABLE = {5: 11, 6: 17, 7: 25, 8: 34, 9: 44, 10: 55, 11: 72, 12: 85}

# Scale exponent for the rigorous integer interval bound (Method 2).
K = 20
SCALE = 10 ** (2 * K)

# Working precision for the Decimal method (>= 50 significant digits).
DECIMAL_PREC = 60


def isqrt_bounds(s):
    """Return integers (lower, upper) with lower/10**K <= sqrt(s) <= upper/10**K,
    exact, using only integer arithmetic (math.isqrt)."""
    scaled = s * SCALE
    lower = math.isqrt(scaled)
    upper = lower if lower * lower == scaled else lower + 1
    return lower, upper


def check_file(path):
    print(f"=== {path} ===")
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  FAIL: could not read/parse file: {e}")
        print()
        return

    if not isinstance(data, dict) or not data:
        print("  FAIL: expected a non-empty JSON object mapping n -> {points: ...}")
        print()
        return

    for key, val in data.items():
        print(f"--- key = {key!r} ---")
        overall_ok = True

        points = None
        if isinstance(val, dict):
            points = val.get("points")
        if points is None:
            print("  FAIL: no 'points' field for this key")
            print()
            continue

        pts = [tuple(int(c) for c in p) for p in points]
        n = len(pts)

        try:
            n_claimed = int(key)
        except ValueError:
            print(f"  FAIL: key {key!r} is not an integer")
            n_claimed = None
            overall_ok = False

        if n_claimed is not None and n != n_claimed:
            print(f"  FAIL: key claims n={n_claimed} but points list has {n} points")
            overall_ok = False

        # --- distinct points check ---
        pts_set = set(pts)
        if len(pts_set) != n:
            print(f"  FAIL: duplicate points found ({n - len(pts_set)} duplicate(s))")
            overall_ok = False

        # --- all pairwise squared distances, exact integers ---
        sq_list = []
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + 1, n):
                xj, yj = pts[j]
                dx = xi - xj
                dy = yi - yj
                sq_list.append(dx * dx + dy * dy)

        m = len(sq_list)
        expected_m = n * (n - 1) // 2
        if m != expected_m:
            print(f"  FAIL: computed {m} pairwise distances, expected n(n-1)/2 = {expected_m}")
            overall_ok = False

        sq_sorted = sorted(sq_list)

        # --- distinct squared distances check ---
        distinct_count = len(set(sq_sorted))
        if distinct_count != m:
            print(f"  FAIL: {m - distinct_count} pair(s) of equal squared distances "
                  f"(i.e. equal actual distances) found")
            overall_ok = False

        # ---------------- Method 1: Decimal, high precision ----------------
        getcontext().prec = DECIMAL_PREC
        dists = [Decimal(s).sqrt() for s in sq_sorted]  # ascending, since sq_sorted is ascending
        if len(dists) >= 2:
            gaps_dec = [dists[i + 1] - dists[i] for i in range(len(dists) - 1)]
            g_dec = min(gaps_dec)
        else:
            g_dec = None
        D_dec = dists[-1] if dists else None
        ratio_dec = (D_dec / g_dec) if (g_dec is not None and g_dec != 0) else None

        # ---------------- Method 2: rigorous integer interval bound ----------------
        bounds = [isqrt_bounds(s) for s in sq_sorted]  # aligned index-for-index with sq_sorted
        lowers = [b[0] for b in bounds]
        uppers = [b[1] for b in bounds]
        if len(bounds) >= 2:
            gap_lower_scaled_list = [lowers[i + 1] - uppers[i] for i in range(len(bounds) - 1)]
            min_gap_lower_scaled = min(gap_lower_scaled_list)
        else:
            min_gap_lower_scaled = None
        D_upper_scaled = uppers[-1] if uppers else None

        if min_gap_lower_scaled is not None and min_gap_lower_scaled > 0:
            rigorous_ratio = Fraction(D_upper_scaled, min_gap_lower_scaled)
        else:
            rigorous_ratio = None
            if min_gap_lower_scaled is not None:
                print(f"  FAIL: rigorous lower bound on g is non-positive "
                      f"(min_gap_lower_scaled={min_gap_lower_scaled}); "
                      f"two distances are equal or too close to separate at K={K}")
                overall_ok = False

        # ---------------- report ----------------
        print(f"  n = {n} (claimed {n_claimed})")
        print(f"  number of pairwise distances = {m} (expected {expected_m})")
        if ratio_dec is not None:
            print(f"  D/g (decimal, {DECIMAL_PREC}-digit precision) = {ratio_dec}")
        else:
            print("  D/g (decimal): N/A (degenerate: fewer than 2 distances, or g=0)")

        if rigorous_ratio is not None:
            disp = Decimal(rigorous_ratio.numerator) / Decimal(rigorous_ratio.denominator)
            print(f"  D/g rigorous upper bound            = {disp}")
        else:
            print("  D/g rigorous upper bound: N/A")

        limit = LIMIT_TABLE.get(n)
        if limit is None:
            print(f"  no LIMIT defined in table for n={n}; skipping PASS/FAIL")
        else:
            bound_ok = (rigorous_ratio is not None) and (rigorous_ratio < limit)
            passed = overall_ok and bound_ok
            status = "PASS" if passed else "FAIL"
            print(f"  LIMIT(n={n}) = {limit}   rigorous upper bound < LIMIT ? "
                  f"{'yes' if bound_ok else 'no'}   ->  {status}")
        print()


def main():
    if len(sys.argv) < 2:
        print("usage: python referee_check.py FILE.json [FILE2.json ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        check_file(path)


if __name__ == "__main__":
    main()
