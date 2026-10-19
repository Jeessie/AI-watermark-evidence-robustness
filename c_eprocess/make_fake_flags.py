"""
Fake flags, for developing the analysis before the real data exists (plan's C9).

Writes the same jsonl format as extract_flags.py, so run_eval.py cannot tell the
difference. Nothing here is a substitute for the real data - the point is that C4/C6/C10
code and plots can be finished and reviewed today, and swapping in real files is a
one-line change.

  python c_eprocess/make_fake_flags.py --out data/fake

Numbers used (change them when the real measurements arrive):
  null green ratio 0.237   (measured on data/human.jsonl, see README)
  wm   green ratio 0.627   (measured on data/wm.jsonl)
  attacked: a rewritten token falls back to Bernoulli(gamma), so
            q_eff = (1 - frac) * q + frac * gamma
  repeated bigrams: tokens drawn from a small Zipf vocabulary, so dedup has something
  to remove (~10-15% of positions), like the real texts.
"""
import argparse
import json
import os

import numpy as np

GAMMA = 0.25
Q_WM = 0.627
Q_NULL = 0.237


def bigrams(n_tokens, vocab, rng, zipf=1.1):
    w = 1.0 / np.arange(1, vocab + 1) ** zipf
    w /= w.sum()
    ids = rng.choice(vocab, size=n_tokens + 1, p=w)   # n_tokens flags need n_tokens+1 ids
    return [(int(ids[i - 1]), int(ids[i])) for i in range(1, n_tokens + 1)]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    g = np.mean([np.mean(r["flags"]) for r in rows])
    keep = np.mean([np.mean(r["first_occurrence"]) for r in rows])
    print(f"  {path}: {len(rows)} texts, green {g:.3f}, kept after dedup {keep:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake")
    ap.add_argument("--n_null", type=int, default=500)
    ap.add_argument("--n_wm", type=int, default=200)
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def make(rate, n, source, frac=None):
        rows = []
        for i in range(n):
            pairs = bigrams(args.tokens, args.vocab, rng)
            seen, keep = set(), []
            for pr in pairs:
                keep.append(pr not in seen)
                seen.add(pr)
            rows.append({"id": f"{source}-{i:04d}", "source": source,
                         "flags": (rng.random(len(pairs)) < rate).astype(int).tolist(),
                         "first_occurrence": keep})
            if frac is not None:
                rows[-1]["attack_frac"] = frac
        return rows

    write(os.path.join(args.out, "flags_human.jsonl"),
          make(Q_NULL, args.n_null, "human"))
    write(os.path.join(args.out, "flags_wm.jsonl"),
          make(Q_WM, args.n_wm, "wm"))

    att = []
    for frac in (0.25, 0.5, 1.0):
        q_eff = (1 - frac) * Q_WM + frac * GAMMA
        att += make(q_eff, args.n_wm, "wm", frac=frac)
    write(os.path.join(args.out, "flags_attacked_wm.jsonl"), att)

    attn = []
    for frac in (0.25, 0.5, 1.0):
        attn += make(Q_NULL, args.n_wm, "human", frac=frac)
    write(os.path.join(args.out, "flags_attacked_human.jsonl"), attn)


if __name__ == "__main__":
    main()
