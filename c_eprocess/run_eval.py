"""
C4 / C6 / C10 numbers straight from flags_*.jsonl (numpy only, no torch).

  python c_eprocess/run_eval.py --null data/flags_human.jsonl \
                                --wm   data/flags_wm.jsonl \
                                --attacked data/flags_attacked_wm.jsonl \
                                --out results/

Produces, in --out:

  fpr.csv           C4: false positive rate of every detector on null texts,
                    dedup on/off, with a Wilson 95% interval (alpha in {0.10,0.05,0.01})
  power_curve.csv   C6: rejection rate vs token count, fixed-length z vs e-processes
  stopping.csv      C6: quantiles of the e-process stopping time (in tokens)
  stops.csv         C6: the same, one line per document (feeds the fig-4 histogram)
  alpha_match.csv   C6: same comparison after matching the ACTUAL false positive rate
                    (the fixed z threshold is calibrated on the null texts, one look at n=600)
  attack.csv        C10: rejection rate by attack_frac x token count, plus L* = the smallest
                    n reaching 90%, or "unreliable", and the raw green ratio

Everything is a plain csv so the plots are somebody else's five lines of matplotlib.
Curves are cached per (file, dedup) because the OG e-process is the expensive part.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eprocess as ep  # noqa: E402

ALL_METHODS = ["nonadapt", "wadapt", "og", "avg", "bet", "betmix"]
LENGTHS = (25, 50, 75, 100, 150, 200, 300, 400, 500, 600)
ALPHAS = (0.10, 0.05, 0.01)
DEFAULT_ALPHA = 0.01


# --------------------------------------------------------------------------- #
def load(path):
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    recs = [r for r in recs if r.get("valid", True)]
    print(f"  {path}: {len(recs)} texts")
    return recs


def pad(M, T):
    """Pad each curve with its last value so all documents have length T.

    Padding is the right thing for "did it reject by time t" questions: a text with fewer
    than t tokens cannot gather more evidence after its last token.
    """
    T = max(T, max(len(m) for m in M))
    return np.array([np.concatenate([m, np.full(T - len(m), m[-1])])[:T] for m in M])


def curves(recs, methods=ALL_METHODS, dedup=True, T=600):
    """M_t for every e-process, plus the running z statistic (raw tokens and deduped)."""
    out = {m: [] for m in methods}
    z_all, z_ded = [], []
    for r in recs:
        flags = np.asarray(r["flags"], dtype=np.float64)
        keep = np.asarray(r["first_occurrence"], dtype=bool)
        f = flags[keep] if dedup else flags
        p = ep.pvalues(f, mode="rand", seed=ep.doc_seed(r["id"]))   # one draw per text
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
        if "betmix" in methods:
            out["betmix"].append(ep.eprocess_bet_mix(f))
        z_all.append(ep.z_curve(flags))
        z_ded.append(ep.z_curve(f))
    res = {k: pad(v, T) for k, v in out.items()}
    res["z"] = pad(z_all, T)
    res["z_dedup"] = pad(z_ded, T)
    return res


class CurveCache:
    """One computation per (file, dedup); every table reuses it."""

    def __init__(self, methods):
        self.methods = methods
        self.cache = {}

    def get(self, path, recs, dedup=True):
        key = (path, dedup)
        if key not in self.cache:
            self.cache[key] = curves(recs, self.methods, dedup)
        return self.cache[key]


def by(M):
    """Running maximum: "did it reject by time t" is a question about max_{s<=t} M_s.

    An e-process is not monotone (the betting and adaptive versions can go down when a token is
    red), so comparing M_t itself would make the detection rate fall as t grows. Ville's
    inequality is about sup_t M_t, so the running maximum is the right statistic.
    """
    return np.maximum.accumulate(np.asarray(M), axis=1)


def wilson(k, n, z=1.96):
    """95% interval for a binomial proportion. Use the upper bound, not the point estimate."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
def table_fpr(files, cc, out_dir):
    rows = []
    for path, recs in files:
        for dedup in (True, False):
            c = cc.get(path, recs, dedup)
            tag = "dedup" if dedup else "raw"
            for m in cc.methods:
                for a in ALPHAS:
                    k = int((c[m].max(1) >= 1.0 / a).sum())
                    lo, hi = wilson(k, len(recs))
                    rows.append([os.path.basename(path), tag, "eprocess_" + m, a,
                                 k, len(recs), round(k / len(recs), 4), round(lo, 4), round(hi, 4)])
            for zname in ("z", "z_dedup"):
                for a in ALPHAS:
                    k = int((c[zname].max(1) > ep.z_threshold(a)).sum())
                    lo, hi = wilson(k, len(recs))
                    rows.append([os.path.basename(path), tag, "peeking_" + zname, a,
                                 k, len(recs), round(k / len(recs), 4), round(lo, 4), round(hi, 4)])
            for n in LENGTHS:                       # valid only at this single length
                for a in ALPHAS:
                    k = int((c["z"][:, n - 1] > ep.z_threshold(a)).sum())
                    lo, hi = wilson(k, len(recs))
                    rows.append([os.path.basename(path), tag, f"fixed_z_n{n}", a,
                                 k, len(recs), round(k / len(recs), 4), round(lo, 4), round(hi, 4)])
    write_csv(os.path.join(out_dir, "fpr.csv"),
              ["file", "dedup", "detector", "alpha", "k", "N", "fpr", "ci_lo", "ci_hi"], rows)


def table_power(files, cc, out_dir):
    rows, stops_rows, per_doc = [], [], []
    for path, recs in files:
        c = cc.get(path, recs, True)
        for m in cc.methods:
            M = c[m]
            for n in LENGTHS:
                rows.append([os.path.basename(path), "eprocess_" + m, n,
                             round(float(np.mean(by(M)[:, n - 1] >= 1.0 / DEFAULT_ALPHA)), 4)])
            for rec, x in zip(recs, M):          # per-document stop time -> fig 4 (histogram)
                st = ep.first_cross(x, DEFAULT_ALPHA)
                per_doc.append([os.path.basename(path), "eprocess_" + m, rec["id"],
                                -1 if st is None else st, len(rec["flags"])])
            got = [s for s in (ep.first_cross(x, DEFAULT_ALPHA) for x in M) if s is not None]
            for q in (10, 25, 50, 75, 90):
                stops_rows.append([os.path.basename(path), "eprocess_" + m, f"p{q}",
                                   int(np.percentile(got, q)) if got else -1])
            stops_rows.append([os.path.basename(path), "eprocess_" + m, "p_no_stop",
                               round(1 - len(got) / len(M), 4)])
        for n in LENGTHS:
            rows.append([os.path.basename(path), "fixed_z", n,
                         round(float(np.mean(c["z"][:, n - 1] > ep.z_threshold(DEFAULT_ALPHA))), 4)])
        rows.append([os.path.basename(path), "peeking_z", 600,
                     round(float(np.mean((c["z"] > ep.z_threshold(DEFAULT_ALPHA)).any(1))), 4)])
    write_csv(os.path.join(out_dir, "power_curve.csv"), ["file", "detector", "n", "tpr"], rows)
    write_csv(os.path.join(out_dir, "stopping.csv"),
              ["file", "detector", "quantile", "value"], stops_rows)
    write_csv(os.path.join(out_dir, "stops.csv"),
              ["file", "detector", "id", "stop", "n_tokens"], per_doc)


def table_alpha_match(null_files, wm_files, cc, out_dir):
    """Same ACTUAL false positive rate for both detectors, then compare power at n=600.

    The fixed-length z-test has no scale-free threshold, but its threshold can be
    calibrated on the null texts: take the (1 - fpr) quantile of z_600 on human text.
    This is the fair comparison TA asked for; report both this and the nominal-alpha one.
    """
    rows = []
    for m in [x for x in ("avg", "bet", "betmix") if x in cc.methods]:
        nulls = null_files[0][1]
        cnull = cc.get(null_files[0][0], nulls, True)
        fpr = float(np.mean(cnull[m].max(1) >= 1.0 / DEFAULT_ALPHA))
        thr = float(np.quantile(cnull["z"][:, -1], 1 - max(fpr, 1e-3)))
        for path, recs in wm_files:
            c = cc.get(path, recs, True)
            rows.append([os.path.basename(path), "eprocess_" + m, round(fpr, 4), round(thr, 4),
                         round(float(np.mean(c[m].max(1) >= 1.0 / DEFAULT_ALPHA)), 4),
                         round(float(np.mean(c["z"][:, -1] > thr)), 4)])
        print(f"  alpha-matched on {m}: actual FPR {fpr:.4f}, z threshold {thr:.3f}")
    write_csv(os.path.join(out_dir, "alpha_match.csv"),
              ["file", "eprocess", "actual_fpr", "z_threshold", "tpr_eprocess", "tpr_z_fixed"],
              rows)


def table_attack(files, cc, out_dir):
    rows = []
    for path, recs in files:
        fracs = sorted({r.get("attack_frac", 0.0) for r in recs})
        for frac in fracs:
            sub = [r for r in recs if r.get("attack_frac", 0.0) == frac]
            if not sub:
                continue
            c = curves(sub, cc.methods, True)
            for m in [x for x in ("avg", "bet", "betmix") if x in cc.methods]:
                M = c[m]
                tprs = []
                for n in LENGTHS:
                    tpr = float(np.mean(by(M)[:, n - 1] >= 1.0 / DEFAULT_ALPHA))
                    tprs.append(tpr)
                    rows.append([os.path.basename(path), frac, "eprocess_" + m, n, round(tpr, 4)])
                rows.append([os.path.basename(path), frac, "eprocess_" + m, "L*",
                             next((n for n, t in zip(LENGTHS, tprs) if t >= 0.9), "unreliable")])
                rows.append([os.path.basename(path), frac, "eprocess_" + m, "green_ratio",
                             round(float(np.mean([np.mean(r["flags"]) for r in sub])), 4)])
            for n in LENGTHS:
                rows.append([os.path.basename(path), frac, "fixed_z", n,
                             round(float(np.mean(c["z"][:, n - 1] > ep.z_threshold(DEFAULT_ALPHA))), 4)])
    write_csv(os.path.join(out_dir, "attack.csv"),
              ["file", "attack_frac", "detector", "n", "tpr"], rows)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", nargs="+", default=None, help="flags_*.jsonl of null texts")
    ap.add_argument("--wm", nargs="+", default=None, help="flags_*.jsonl of watermarked texts")
    ap.add_argument("--attacked", nargs="+", default=None, help="flags of attacked texts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default=None, help="comma list of tables: fpr,power,match,attack")
    ap.add_argument("--methods", default=",".join(ALL_METHODS),
                    help="subset of nonadapt,wadapt,og,avg,bet (drop og/avg to go ~5x faster)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    only = set(args.only.split(",")) if args.only else {"fpr", "power", "match", "attack"}
    methods = [m for m in args.methods.split(",") if m]
    cc = CurveCache(methods)

    null_files = [(p, load(p)) for p in (args.null or [])]
    wm_files = [(p, load(p)) for p in (args.wm or [])]
    att_files = [(p, load(p)) for p in (args.attacked or [])]

    if "fpr" in only and null_files:
        print("C4: false positive rate on null texts")
        table_fpr(null_files, cc, args.out)
    if "power" in only and wm_files:
        print("C6: power vs token count, stopping times")
        table_power(wm_files, cc, args.out)
    if "match" in only and null_files and wm_files:
        print("C6: matched actual FPR")
        table_alpha_match(null_files, wm_files, cc, args.out)
    if "attack" in only and att_files:
        print("C10: attacked texts")
        table_attack(att_files, cc, args.out)


if __name__ == "__main__":
    main()
