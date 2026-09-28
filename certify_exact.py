"""Rigorous integer-only certificate for a planar configuration (Erdős #670).

Input: integer coordinates P (any unit).  All squared distances s are integers.  For consecutive sorted squared distances
a < b, sqrt(b) - sqrt(a) = (b - a) / (sqrt(a) + sqrt(b)) >= (b - a) / (ceilsqrt(a) + ceilsqrt(b)), so
g_low = min over consecutive pairs of that fraction is a rigorous lower bound for the minimum gap g, and ceilsqrt(max s)
is a rigorous upper bound for the diameter D.  The configuration scaled by 1/g has all distances at least 1 apart and
diameter D/g <= ceilsqrt(max s) / g_low.  Uses only Python integers and fractions (no floating point).

Usage: python certify_exact.py FILE.json [KEY]   (FILE maps keys to {"points": [[x, y], ...]})
"""
import sys, json
from fractions import Fraction
from math import isqrt

def ceilsqrt(s):
    r = isqrt(s)
    return r if r * r == s else r + 1

def certify(P):
    n = len(P)
    sq = sorted((P[i][0] - P[j][0]) ** 2 + (P[i][1] - P[j][1]) ** 2 for i in range(n) for j in range(i + 1, n))
    assert len(set(sq)) == len(sq), "two equal distances"
    g_low = min(Fraction(b - a, ceilsqrt(a) + ceilsqrt(b)) for a, b in zip(sq, sq[1:]))
    D_up = ceilsqrt(sq[-1])
    return len(sq), g_low, D_up, Fraction(D_up) / g_low

if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    keys = sys.argv[2:] or sorted(data, key=lambda k: int(k))
    for k in keys:
        m, g_low, D_up, ratio_up = certify(data[k]["points"])
        # print the upper bound rounded UP at 6 decimals
        num, den = ratio_up.numerator * 10**6, ratio_up.denominator
        up6 = -(-num // den)
        print(f"n={k}: {m} distinct distances; min gap >= {float(g_low):.6f}, diameter <= {D_up} (units); "
              f"certified diameter D/g <= {up6 // 10**6}.{up6 % 10**6:06d}")
