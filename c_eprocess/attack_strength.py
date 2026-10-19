"""
How strong was the attack, objectively? (feeds the x-axis of the length/attack figures)

"light / strong" is a prompt style, and "1.5B / 7B" is an attacker size -- neither is a
measurement. Without a number, a reader cannot tell whether a 7B rewrite is really harsher
than a 1.5B one, or whether the residual green ratio came from the attack or from the
sampling noise. So for every attacked text we report how much of the original token
sequence survived:

  token_kept   fraction of the original token positions that survived (LCS / len(orig))
               1.0 = nothing changed, 0.30 = two thirds of the tokens were replaced
  char_ratio   len(new) / len(orig) -- catches "rewrites" that just delete or pad text
  unigram_new  fraction of the new tokens that never appear in the original

  python c_eprocess/attack_strength.py data/attacked_wm.jsonl data/attacked_api.jsonl \
         --orig data/wm.jsonl --out results/real/attack_strength.csv

B group's records already carry `token_edit_ratio`; for them this script reuses that number
instead of recomputing, so both arms are reported on the same scale.

Then join it with attack.csv (same `orig_id`/`attack_frac`) when plotting, or just read the
per-attacker summary printed at the end -- that table is what belongs in the report next to
the green ratios.
"""
import argparse
import csv
import difflib
import json
import os
import statistics

import numpy as np


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attacked", nargs="+", help="attacked jsonl files (they carry orig_id)")
    ap.add_argument("--orig", default="data/wm.jsonl", help="jsonl with the original token_ids")
    ap.add_argument("--tokenizer", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="results/real/attack_strength.csv")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    orig = {}
    for r in load(args.orig):
        orig[r["id"]] = (r["token_ids"], len(r.get("text", "")))

    rows, summary = [], {}
    for path in args.attacked:
        # 我们自己的攻击带 orig_id；B 组交付的数据直接把 id 当原始 id 用
        recs = [r for r in load(path)
                if r.get("valid", True) and (r.get("orig_id") or r["id"]) in orig]
        print(f"{path}: {len(recs)} texts, attacker {recs[0].get('attacker', '?') if recs else '?'}")
        per_frac = {}
        for r in recs:
            oid = r.get("orig_id") or r["id"]
            a, orig_chars = orig[oid]
            b = tok(r["text"], add_special_tokens=False)["input_ids"]
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            kept = sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(a))
            if "token_edit_ratio" in r:                 # B 组已给出同口径指标
                kept = 1.0 - float(r["token_edit_ratio"])
            new = 1.0 - len(set(b) & set(a)) / max(1, len(set(b)))
            row = {"file": os.path.basename(path),
                   "attacker": r.get("attacker") or "bgroup:" + str(
                       (r.get("attack_config") or {}).get("model", "qwen2.5-1.5b")),
                   "id": r["id"], "orig_id": oid, "attack_frac": r["attack_frac"],
                   "attack_strength": r.get("attack_strength", ""), "passes": r.get("passes", 1),
                   "token_kept": round(kept, 4),
                   "char_ratio": round(len(r["text"]) / max(1, orig_chars), 4),
                   "unigram_new": round(new, 4), "green_ratio": round(float(np.mean(r.get("flags", [0]))), 4)
                   if r.get("flags") else ""}
            rows.append(row)
            per_frac.setdefault(r["attack_frac"], []).append(row)
        for frac, rs in sorted(per_frac.items()):
            summary[(os.path.basename(path), frac)] = (
                statistics.mean(r["token_kept"] for r in rs),
                statistics.mean(r["unigram_new"] for r in rs), len(rs))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)\n")

    print(f"{'attacker file':28s} {'frac':>5s} {'token_kept':>11s} {'unigram_new':>12s} {'n':>4s}")
    for (f, frac), (kept, new, n) in summary.items():
        print(f"{f:28s} {frac:5.2f} {kept:11.3f} {new:12.3f} {n:4d}")


if __name__ == "__main__":
    main()
