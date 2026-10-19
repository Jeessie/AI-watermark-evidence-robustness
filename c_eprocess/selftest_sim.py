"""
C3 + C4 without the data: check the detectors on synthetic green flags.

No torch, no LLM. This is the step the plan calls "在模拟数据上验证代码写得对": if the
e-process false positive rate is not <= alpha here, and the peeking z-test is not clearly
above alpha, then something is wrong with the code and nothing downstream is worth doing.

  H0 (human text):    flag ~ Bernoulli(gamma = 0.25)
  H1 (watermarked):   flag ~ Bernoulli(gamma + delta*gamma*(1-gamma) = 0.625)

The H1 rate is what KGW's green-list bias implies, and it matches what we measure on
data/wm.jsonl (green ratio 0.627).

  python c_eprocess/selftest_sim.py                 # 2000 x 600, ~4 min
  python c_eprocess/selftest_sim.py --docs 10000    # the plan's full size, ~25 min

For each number we report a Wilson 95% interval, because at alpha = 1% with 500-2000
documents the point estimate is very noisy - that interval is exactly what C4 needs.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eprocess as ep  # noqa: E402

LENGTHS = (100, 200, 400, 600)
EPS = ["nonadapt", "wadapt", "og", "avg", "bet", "betmix"]
ALPHA = 0.01
SEED = 0


# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def simulate(rate, n_docs, n_tokens, seed=SEED):
    rng = np.random.default_rng(seed)
    return (rng.random((n_docs, n_tokens)) < rate).astype(np.float64)


def run(flags, methods=EPS):
    out = {"z": []}
    out.update({m: [] for m in methods})
    for i, f in enumerate(flags):
        p = ep.pvalues(f, mode="rand", seed=ep.doc_seed(i))   # one draw per text
        out["z"].append(ep.z_curve(f))
        if "nonadapt" in methods:
            out["nonadapt"].append(ep.eprocess_nonadaptive(p))
        if "wadapt" in methods:
            out["wadapt"].append(ep.eprocess_weight_adaptive(p))
        if "og" in methods:
            out["og"].append(ep.eprocess_og(p))
        if "avg" in methods:
            out["avg"].append(ep.eprocess_average(p))
        if "bet" in methods:
            out["bet"].append(ep.eprocess_bet(f))
        if "betmix" in methods:                 # 主检测器也要进自检
            out["betmix"].append(ep.eprocess_bet_mix(f))
    return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
def check_og_calibrator():
    """The OG calibrator must be a decreasing density on [0,1]: positive, block slopes
    non-increasing, integral 1 minus the atom at p=0, and its CDF must majorise the
    empirical CDF (that is what "least concave majorant" means)."""
    p = ep.pvalues(simulate(0.625, 1, 300)[0], mode="rand", seed=SEED)
    f = ep.og_calibrator_values(p)
    xs, ws = ep._unique_weights(np.concatenate([[0.0], p[:200], [1.0]]),
                               np.concatenate([[0.5], np.ones(200), [0.5]]))
    left, right, slope = ep._lcm_density(xs, ws)
    width = right - left
    integ = float(np.dot(slope, width))
    cdf_emp = np.cumsum(ws) / ws.sum()
    y0 = cdf_emp[0]                       # atom at p = 0, not part of the density
    # fitted CDF at every data value, read off the block that contains it
    k = np.searchsorted(left, xs, side="right") - 1
    prefix = np.concatenate(([0.0], np.cumsum(slope * width)[:-1]))
    cdf_fit = y0 + prefix[k] + slope[k] * (xs - left[k])
    print("=== OG calibrator sanity (300 synthetic watermarked p-values, t = 1..300)")
    print(f"   all f_t > 0                        : {bool((f > 0).all())}")
    print(f"   block slopes non-increasing        : {bool(np.all(np.diff(slope) <= 1e-12))}")
    print(f"   integral over [0,1]                : {integ:.6f}   "
          f"(= 1 - atom at p=0 = {1 - y0:.6f})")
    print(f"   fitted CDF (+atom) >= empirical CDF: {bool(np.all(cdf_fit >= cdf_emp - 1e-12))}")
    print(f"   f_1, f_2 (eq13 starts at 0.5)      : {f[0]:.3f}, {f[1]:.3f}")
    print()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--skip_og", action="store_true", help="skip the slow OG/avg methods")
    args = ap.parse_args()
    methods = [m for m in EPS if not (args.skip_og and m in ("og", "avg"))]

    check_og_calibrator()

    z = ep.z_threshold(ALPHA)
    t0 = time.time()
    h0 = simulate(ep.GAMMA, args.docs, args.tokens)
    h1 = simulate(ep.GAMMA + ep.DELTA * ep.GAMMA * (1 - ep.GAMMA), args.docs, args.tokens)
    r0, r1 = run(h0, methods), run(h1, methods)
    dt = time.time() - t0
    print(f"=== {args.docs} docs x {args.tokens} tokens, H0 and H1, in {dt:.0f}s "
          f"({dt/(2*args.docs*args.tokens)*1e3:.3f} ms/token, OG dominates)\n")

    n = args.docs
    print(f"=== C3/C4: false positive rate on H0 (human text), nominal alpha = {ALPHA}")
    print("   e-process: P(sup_t M_t >= 1/alpha) <= alpha for every stopping rule")
    print("   fixed z  : one look at a pre-declared n (valid), z > %.3f\n" % z)
    print(f"{'detector':22s} {'FPR':>7s}  90% of the Wilson 95% CI upper bound "
          f"{'k/N':>10s}")
    rows = []
    for m in methods:
        M = r0[m]
        k = int((M.max(1) >= 1.0 / ALPHA).sum())
        lo, hi = wilson(k, n)
        rows.append((f"e-process {m}", k, n))
        print(f"{'e-process ' + m:22s} {k/n:7.4f}  [{lo:.4f}, {hi:.4f}]  {k:5d}/{n}")
    k = int((r0["z"].max(1) > z).sum())
    lo, hi = wilson(k, n)
    print(f"{'peeking z (INVALID)':22s} {k/n:7.4f}  [{lo:.4f}, {hi:.4f}]  {k:5d}/{n}"
          "   <-- FPR explodes")
    for t in LENGTHS:
        k = int((r0["z"][:, t - 1] > z).sum())
        lo, hi = wilson(k, n)
        print(f"{'fixed z, n=%d' % t:22s} {k/n:7.4f}  [{lo:.4f}, {hi:.4f}]  {k:5d}/{n}"
              "   <-- only valid at this t")
    print()

    print(f"=== C6: power on H1 (watermarked text) and stopping time, alpha = {ALPHA}")
    print(f"{'detector':22s} " + " ".join(f"t={t:<5d}" for t in LENGTHS)
          + f" {'median stop':>11s} {'P(stop<=600)':>12s}")
    for m in methods:
        M = r1[m]
        row = [np.mean(M[:, t - 1] >= 1.0 / ALPHA) for t in LENGTHS]
        stops = [s for s in (ep.first_cross(x, ALPHA) for x in M) if s is not None]
        med = int(np.median(stops)) if stops else -1
        print(f"{'e-process ' + m:22s} " + " ".join(f"{v:7.3f}" for v in row)
              + f" {med:11d} {len(stops)/n:12.3f}")
    z1 = r1["z"]
    print(f"{'peeking z (INVALID)':22s} " + " ".join(
        f"{np.mean(z1[:, t-1] > z):7.3f}" for t in LENGTHS)
        + f" {'-':>11s} {np.mean((z1 > z).any(1)):12.3f}")
    print(f"{'fixed z, one look':22s} " + " ".join(
        f"{np.mean(z1[:, t-1] > z):7.3f}" for t in LENGTHS) + f" {'-':>11s} {'-':>12s}")


if __name__ == "__main__":
    main()
