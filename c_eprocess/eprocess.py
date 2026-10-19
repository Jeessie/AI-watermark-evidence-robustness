"""
C group: online (sequential) KGW watermark detection.

Three detectors, all driven by the same per-token green-flag sequence:

  1. fixed-length z-test     the KGW statistic, decided once at a fixed token count
  2. peeking z-test          the same z statistic but checked after every token
                             (this is what an online deployment naturally does;
                              it does NOT control the false positive rate)
  3. e-process               Su, Wang & Zhao (2026, arXiv 2602.14286): a nonnegative
                             martingale under H0, reject when M_t >= 1/alpha.
                             Ville's inequality => Type I error <= alpha under ANY
                             stopping rule, so "peeking" is free.

Everything here is numpy only. Flags are produced by extract_flags.py (needs torch).

-------------------------------------------------------------------------------
Why KGW needs a p-value step before the e-process
-------------------------------------------------------------------------------
The e-process machinery in the paper is stated for a pivotal statistic Y_t that is

  (i)  exactly U(0,1) and independent of the past under H0, and
  (ii) stochastically smaller (super-uniform) under H1.

KGW does NOT hand you such a Y. KGW hands you a green flag

  flag_t = 1{ w_t in greenlist(w_{t-1}) },   P(flag_t = 1) = gamma under H0
                                             P(flag_t = 1) > gamma under H1

so under H0 flag_t is Bernoulli(gamma), not U(0,1). Two valid conversions to a
p-value p_t (small p = evidence for the watermark), p_t = 1 - Y_t:

  'rand'  p = gamma * U            if green     (U ~ U(0,1) fresh, seeded)
          p = gamma + (1-gamma) * U if red
          -> exactly U(0,1) under H0, continuous.  Keeps the "how green" information.

  'disc'  p = gamma if green, p = 1 if red
          -> valid but conservative (P(p<=gamma)=gamma). Never rewards a red token
             with anything but 1, and loses the randomization.

Both satisfy E_H0[p <= u] <= u, which is all Theorem 1 needs. See docs/C_report.md §3.1
for the caveat that KGW flags are not exactly independent under H0 (shared green lists,
repeated n-grams) - that is what the human-text FPR check and the dedup option are for.
"""
from __future__ import annotations

import bisect
import math
from statistics import NormalDist

import numpy as np

# KGW hyper-parameters (configs/KGW_g025.json)
GAMMA = 0.25
DELTA = 2.0


# --------------------------------------------------------------------------- #
# flags -> p-values
# --------------------------------------------------------------------------- #
def doc_seed(key):
    """Stable per-document seed for the randomised p-value.

    The randomisation is part of the test, so every text needs its own draw. Reusing one
    seed for a whole file makes the texts share the same uniform sequence: in practice a
    few unlucky texts then all cross 1/alpha at the same time (we saw M_t = 103.0 on two
    different human texts), which inflates the measured false positive rate without any
    watermark being present. Same idea as re-randomising a randomised test per subject.
    """
    import zlib
    return zlib.crc32(str(key).encode())


def pvalues(flags, gamma=GAMMA, mode="rand", seed=0):
    """Per-token p-value from KGW green flags. flags: array of 0/1.

    seed: pass `doc_seed(text_id)` -- one draw per text, not one draw per file.
    """
    flags = np.asarray(flags, dtype=np.float64)
    if mode == "disc":
        return np.where(flags > 0.5, gamma, 1.0)
    if mode == "rand":
        u = np.random.default_rng(seed).random(flags.size)
        return np.where(flags > 0.5, gamma * u, gamma + (1.0 - gamma) * u)
    raise ValueError(f"unknown mode {mode!r}")


def dedup_mask(pairs):
    """True at the first occurrence of each (previous token, current token) pair.

    KGW's flag is a deterministic function of that pair, so a repeated bigram
    ("of the") is the same coin flip over and over: it adds no information and it
    correlates the tokens, which is exactly what Assumptions 1/4 do not want.
    """
    seen = set()
    keep = np.zeros(len(pairs), dtype=bool)
    for i, pair in enumerate(pairs):
        if pair not in seen:
            seen.add(pair)
            keep[i] = True
    return keep


# --------------------------------------------------------------------------- #
# calibrators and e-processes (paper Section 4)
# --------------------------------------------------------------------------- #
def cal_log(p):
    """g(p) = -log p, the calibrator the paper recommends in practice."""
    return -np.log(np.clip(p, 1e-300, 1.0))


def cal_sqrt(p):
    """g(p) = p^{-1/2} - 1 (another calibrator, for sensitivity checks)."""
    return np.clip(p, 1e-300, 1.0) ** -0.5 - 1.0


CALIBRATORS = {"log": cal_log, "sqrt": cal_sqrt}


def _adaptive_mixture(terms, lam_grid):
    """E_t = 1 - lam_t + lam_t * terms_t with lam_t maximising the past log-sum.

    This is eq. (9)/(11) of the paper: predictably chosen mixing weight, so the
    resulting product is still an e-process (Theorem 1).
    """
    n = terms.size
    logM = np.zeros(n)
    S = np.zeros(lam_grid.size)  # cumulative log-likelihood per grid point, past only
    logM_run = 0.0
    for t in range(n):
        lam = lam_grid[int(np.argmax(S))] if t > 0 else 0.0
        e = 1.0 - lam + lam * terms[t]
        logM_run += math.log(e)
        logM[t] = logM_run
        S = S + np.log(1.0 - lam_grid + lam_grid * terms[t])
    return np.exp(logM)


def _mixture_e(x, lam_grid):
    """M_t = mean_j prod_{s<=t} (1 + lam_j x_s): the mixture over lambda.

    An average of e-processes is an e-process, so Ville still applies; unlike the
    follow-the-leader rule in _adaptive_mixture it does not commit to one lambda, and it
    dominates the best fixed lambda up to a log(#grid) additive cost. On a weak signal
    (fully paraphrased text) that gap is large: FTL 49% vs mixture 77% detection at 650
    tokens, same texts.
    """
    L = np.zeros(lam_grid.size)                 # log wealth of each fixed-lambda expert
    lp = -math.log(lam_grid.size)               # uniform prior
    out = np.empty(x.size)
    for t in range(x.size):
        L += np.log1p(lam_grid * x[t])
        m = float(np.max(L + lp))
        out[t] = math.exp(m + math.log(float(np.sum(np.exp(L + lp - m)))))
    return out


def eprocess_bet_mix(flags, gamma=GAMMA, n_grid=201):
    """Mixture (universal portfolio) version of the betting e-process."""
    lam = np.linspace(0.0, 1.0 / gamma, n_grid, endpoint=False)[1:]   # keeps 1 + lam*(-gamma) > 0
    return _mixture_e(np.asarray(flags, dtype=np.float64) - gamma, lam)


def eprocess_wadapt_mix(p, cal="log", lam_max=0.5, n_grid=201):
    """Mixture version of eq. (11): fixed calibrator, average over lambda instead of FTL."""
    return _mixture_e(CALIBRATORS[cal](p) - 1.0, np.linspace(0.0, lam_max, n_grid))


def eprocess_nonadaptive(p, cal="log", lam=0.1):
    """Eq. (8): fixed lambda, fixed calibrator."""
    return np.exp(np.cumsum(np.log(1.0 - lam + lam * CALIBRATORS[cal](p))))


def eprocess_weight_adaptive(p, cal="log", lam_max=0.5, n_grid=201):
    """Eq. (11): fixed calibrator g = -log, predictable lambda_t from eq. (9)."""
    return _adaptive_mixture(CALIBRATORS[cal](p), np.linspace(0.0, lam_max, n_grid))


def eprocess_bet(flags, gamma=GAMMA, n_grid=201):
    """Betting / Kelly e-process on the Bernoulli green flag.

    E_t = 1 + lam_t (flag_t - gamma),  lam_t in [0, 1/gamma]  => E_t >= 0 and
    E_H0[E_t | past] = 1 exactly, so M_t = prod E_t is an e-process. This is the
    tightest of the four for KGW because it never wastes a red token on E_t = 1.
    """
    x = np.asarray(flags, dtype=np.float64) - gamma
    lam_grid = np.linspace(0.0, 1.0 / gamma, n_grid, endpoint=False)[1:]  # keep E_t > 0
    n = x.size
    logM = np.zeros(n)
    S = np.zeros(lam_grid.size)
    run = 0.0
    for t in range(n):
        lam = lam_grid[int(np.argmax(S))] if t > 0 else 0.0
        e = 1.0 + lam * x[t]
        run += math.log(e)
        logM[t] = run
        S = S + np.log(1.0 + lam_grid * x[t])
    return np.exp(logM)


def _lcm_density(xs, ws):
    """Grenander estimator: derivative of the least concave majorant of the EDF.

    xs: sorted UNIQUE values, xs[0] == 0 required (add a zero-weight point if the data
    does not start at 0); ws: their weights. The diagram is (x_j, F_j) with
    F_j = cumulative weight / total, so a block spanning breakpoints j..m has

        f = (F_m - F_j) / (x_m - x_j)

    and pooling enforces the block slopes non-increasing in x, which is the estimator of
    a decreasing density. Mass sitting exactly at x=0 (an atom) is not spread over the
    first block: the fitted density then integrates to 1 - F(0) <= 1, still a calibrator.

    Returns (block_left, block_right, slope).
    """
    total = float(ws.sum())
    y = np.cumsum(ws) / total            # y[j] = F(x_j)
    m = xs.size
    bs = []
    for j in range(m - 1):               # atoms are the SEGMENTS between adjacent points
        bs.append([j, j + 1])
        while len(bs) >= 2:
            s2, e2 = bs[-1]
            s1, e1 = bs[-2]
            num2 = y[e2] - y[s2]
            num1 = y[e1] - y[s1]
            # violation iff slope(previous block) < slope(last block), cross-multiplied
            if num1 * (xs[e2] - xs[s2]) < num2 * (xs[e1] - xs[s1]):
                bs.pop()
                bs[-1][1] = e2
            else:
                break
    bl = np.array([b[0] for b in bs])
    br = np.array([b[1] for b in bs])
    return xs[bl], xs[br], (y[br] - y[bl]) / (xs[br] - xs[bl])


def _unique_weights(v, w=None):
    """Group equal values, summing their weights (duplicate p-values must be pooled)."""
    v = np.asarray(v, dtype=np.float64)
    if w is None:
        w = np.ones(v.size)
    u, inv = np.unique(v, return_inverse=True)
    ww = np.zeros(u.size)
    np.add.at(ww, inv, w)
    return u, ww


def og_calibrator_values(p, variant="eq13"):
    """f_t^OG(p_t) for t = 1..n: the paper's eq. (12)/(13) evaluated online.

    variant="eq13": half the pseudo-mass at p=0 and half at p=1 (the paper's choice for
                    its experiments; less conservative). Note f_1 = 0.5, so M_1 = 0.5 -
                    one token of burn-in, negligible over hundreds of tokens.
    variant="eq12": one pseudo-observation at p=1 (its Y_0 = 0), f_1 = 1 exactly.

    Only the past p-values are used, so f_t is predictable (f_t in Phi_t) and
    M_t = prod_t f_t(p_t) is an e-process. Implemented as insert + refit so each step
    costs O(t) arithmetic: the sorted past values and their weights are plain lists.

    The virtual zero-weight point at p=0 is what lets the leftmost block start at 0.
    """
    n = int(p.size)
    out = np.empty(n)
    if variant == "eq13":
        vals, wts = [0.0, 1.0], [0.5, 0.5]
    elif variant == "eq12":
        vals, wts = [0.0, 1.0], [0.0, 1.0]
    else:
        raise ValueError(variant)
    p = np.asarray(p, dtype=np.float64)

    for i in range(n):
        m = len(vals)
        tot = float(sum(wts))
        cum = [0.0] * m                  # cumulative F, plain Python to keep the refit cheap
        acc = 0.0
        for j in range(m):
            acc += wts[j]
            cum[j] = acc / tot
        bs, be = [0] * m, [0] * m        # PAVA over the m-1 segments
        top = 0
        for j in range(m - 1):
            bs[top], be[top] = j, j + 1
            top += 1
            while top >= 2:
                s2, e2 = bs[top - 1], be[top - 1]
                s1, e1 = bs[top - 2], be[top - 2]
                if (cum[e1] - cum[s1]) * (vals[e2] - vals[s2]) <                    (cum[e2] - cum[s2]) * (vals[e1] - vals[s1]):
                    be[top - 2] = e2
                    top -= 1
                else:
                    break
        pi = p[i]
        k = 0                            # block holding p[i]
        for j in range(top):
            if vals[bs[j]] <= pi:
                k = j
            else:
                break
        s, e = bs[k], be[k]
        out[i] = (cum[e] - cum[s]) / (vals[e] - vals[s])
        pos = bisect.bisect_left(vals, pi)          # insert p[i] for the next step
        if pos < m and vals[pos] == pi:
            wts[pos] += 1.0
        else:
            vals.insert(pos, pi)
            wts.insert(pos, 1.0)
    return out


def eprocess_og(p, variant="eq13"):
    """Eq. (14): M_t = prod f_t^OG(p_t)."""
    f = og_calibrator_values(p, variant=variant)
    return np.exp(np.cumsum(np.log(np.clip(f, 1e-300, None))))


def eprocess_average(p, variant="eq13"):
    """Eq. (15): arithmetic average of the weight-adaptive and the OG e-process."""
    return 0.5 * eprocess_weight_adaptive(p) + 0.5 * eprocess_og(p, variant=variant)


# --------------------------------------------------------------------------- #
# the two z-tests (C's baseline / strawman)
# --------------------------------------------------------------------------- #
def z_curve(flags, gamma=GAMMA):
    """Running KGW z-statistic: (G_t - gamma t) / sqrt(t gamma (1-gamma))."""
    f = np.asarray(flags, dtype=np.float64)
    t = np.arange(1, f.size + 1)
    G = np.cumsum(f)
    return (G - gamma * t) / np.sqrt(t * gamma * (1.0 - gamma))


def z_fixed(flags, gamma=GAMMA, n=None):
    """KGW z score at a fixed length n (default: all tokens available)."""
    f = np.asarray(flags, dtype=np.float64)
    if n is not None:
        f = f[:n]
    n = f.size
    return (f.sum() - gamma * n) / math.sqrt(n * gamma * (1.0 - gamma))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# public API asked for in the plan (C2): same three calls, same return convention
#   stop position (1-indexed) or None
# --------------------------------------------------------------------------- #
def z_threshold(alpha):
    """One-sided normal quantile: the z-test rejects for z > z_{1-alpha}."""
    return NormalDist().inv_cdf(1.0 - alpha)


def fixed_z(flags, n, alpha=0.01):
    """Look once, at a pre-declared length n. Returns True (watermark) / False."""
    return bool(z_fixed(flags, n=n) > z_threshold(alpha))


def peeking_z(flags, alpha=0.01):
    """Check the z-test after EVERY token (invalid). Returns first crossing or None."""
    return first_cross_upper(z_curve(flags), z_threshold(alpha))


def eprocess(flags, alpha=0.01, dedup=True, method="avg", mode="rand", seed=0,
             first_occ=None, pairs=None, variant="eq13"):
    """Sequential test: read tokens one by one, stop when M_t >= 1/alpha.

    method: 'avg' (paper's recommendation, eq.15) | 'wadapt' (eq.11) | 'og' (eq.14)
            | 'nonadapt' (eq.8) | 'bet' (Bernoulli Kelly betting, tightest for KGW)
            | 'betmix' (betting averaged over lambda -- best on weak signals)
            | 'wmix' (eq.11 with the lambda average instead of follow-the-leader)
    dedup : keep only the first occurrence of each (previous token, current token) pair.
            Pass first_occ (the mask from flags_*.jsonl) or pairs (the pairs themselves);
            with neither, dedup is silently off.
    Returns the stopping position measured in TOKENS of the original sequence, or None.
    """
    flags = np.asarray(flags, dtype=np.float64)
    keep = None
    if dedup:
        if first_occ is not None:
            keep = np.asarray(first_occ, dtype=bool)
        elif pairs is not None:
            keep = dedup_mask(pairs)
    if keep is not None:
        pos = np.flatnonzero(keep)
    else:
        pos = np.arange(flags.size)
    f = flags[pos]
    p = pvalues(f, mode=mode, seed=seed)
    if method == "avg":
        M = eprocess_average(p, variant=variant)
    elif method == "wadapt":
        M = eprocess_weight_adaptive(p)
    elif method == "og":
        M = eprocess_og(p, variant=variant)
    elif method == "nonadapt":
        M = eprocess_nonadaptive(p)
    elif method == "bet":
        M = eprocess_bet(f)
    elif method == "betmix":
        M = eprocess_bet_mix(f)
    elif method == "wmix":
        M = eprocess_wadapt_mix(p)
    else:
        raise ValueError(method)
    j = first_cross(M, alpha)
    return None if j is None else int(pos[j - 1]) + 1


def first_cross(M, alpha):
    """First t (1-indexed) with M_t >= 1/alpha, or None."""
    idx = np.flatnonzero(np.asarray(M) >= 1.0 / alpha)
    return int(idx[0]) + 1 if idx.size else None


def first_cross_upper(stat, thresh):
    """First t (1-indexed) with stat_t > thresh, or None. Used for peeking z."""
    idx = np.flatnonzero(np.asarray(stat) > thresh)
    return int(idx[0]) + 1 if idx.size else None
