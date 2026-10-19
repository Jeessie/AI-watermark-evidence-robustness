"""
C1: per-token KGW green flags for every text, as a cached artefact.

Needs torch + MarkLLM (run it on CARC or anywhere MarkLLM is installed); everything
downstream (eprocess.py, run_eval.py) is numpy only and reads the jsonl written here.

  python c_eprocess/extract_flags.py data/wm.jsonl data/human.jsonl
  python c_eprocess/extract_flags.py data/attacked_wm.jsonl data/attacked_human.jsonl

  # attacked texts without token_ids (our own attack arms) need --tokenizer:
  python c_eprocess/extract_flags.py data/attacked_api.jsonl --tokenizer models/Qwen2.5-1.5B-Instruct \
         --out data/flags_attacked_api.jsonl

  # another key (multi-key FPR sweep):
  python c_eprocess/extract_flags.py data/human.jsonl --hash_key 15485929

Output (one line per text):

  {"id", "source", "flags", "first_occurrence", "attack_frac"?, "valid"?,
   "actual_attack_frac"?, "sim"?, "token_edit_ratio"?, "attacker"?, "orig_id"?}

  only valid=true records are scored; resume is keyed by (id, attack_frac), so several
  attack levels of the same id can coexist in one file

  flags[i]           1 if token i is in the green list of the previous token, else 0
  first_occurrence[i] True if (w_{i-1}, w_i) has not appeared earlier in this text
                      -> the dedup mask. KGW's flag is a deterministic function of that
                      pair, so a repeated bigram is the same coin flip counted twice.

Things that are easy to get wrong (also in README):
  * the first generated token is scored with the LAST PROMPT token as its prefix, so we
    feed prompt_ids[-1:] + token_ids and drop the prefix position;
  * vocab_size must be model.config.vocab_size (151936), not len(tokenizer);
  * patch_kgw_cpu() before scoring, or green lists differ between GPUs;
  * attacked texts have only `text`, so pass --tokenizer and they are re-tokenized (the original
    token_ids are gone; decode -> encode round trips differ by a few tokens, README warns about it).

Speed: the green list depends only on the previous token, so we group positions by
previous token and build each green list once (that is what makes this CPU-fast).
"""
import argparse
import collections
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kgw_cpu import patch_kgw_cpu  # noqa: E402


def load_kgw(markllm_dir, wm_config, vocab_size, device, hash_key=None):
    sys.path.insert(0, os.path.abspath(markllm_dir))
    from watermark.kgw.kgw import KGW, KGWConfig
    from utils.transformers_config import TransformersConfig

    tc = TransformersConfig(model=None, tokenizer=None, vocab_size=vocab_size, device=device)
    kgw = KGW(KGWConfig(os.path.abspath(wm_config), tc))
    if hash_key is not None:            # C4's multi-key sweep: same text, different key
        kgw.utils.config.hash_key = hash_key
    return patch_kgw_cpu(kgw)


def flags_for(kgw, seqs, vocab_size, device):
    """seqs: list of token-id lists, each already prefixed with the prompt's last token.

    Returns flags[i] with len(flags[i]) == len(seqs[i]) - 1.
    """
    by_prev = collections.defaultdict(list)
    for k, s in enumerate(seqs):
        for i in range(1, len(s)):
            by_prev[s[i - 1]].append((k, i, s[i]))
    flags = [[0] * (len(s) - 1) for s in seqs]
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for prev, items in by_prev.items():
        mask.zero_()
        mask[kgw.utils.get_greenlist_ids(torch.tensor([prev], device=device))] = True
        hits = mask[torch.tensor([c for _, _, c in items], device=device)].tolist()
        for (k, i, _), h in zip(items, hits):
            flags[k][i - 1] = int(h)
    return flags


def seq_of(r, tok):
    """token id sequence whose first element seeds the green list of the first scored token."""
    if r.get("token_ids"):
        return r["prompt_ids"][-1:] + r["token_ids"]
    if tok is None:
        raise SystemExit(f"{r['id']}: no token_ids and no --tokenizer given")
    return r["prompt_ids"][-1:] + tok(r["text"], add_special_tokens=False)["input_ids"]


def first_occurrence(seqs):
    """Mark the first appearance of each (previous token, current token) pair per text."""
    out = []
    for s in seqs:
        seen, keep = set(), []
        for i in range(1, len(s)):
            pair = (s[i - 1], s[i])
            keep.append(pair not in seen)
            seen.add(pair)
        out.append(keep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="jsonl with prompt_ids and token_ids")
    ap.add_argument("--out", default=None, help="default: data/flags_<basename>.jsonl")
    ap.add_argument("--markllm_dir", default="MarkLLM")
    ap.add_argument("--wm_config", default="configs/KGW_g025.json")
    ap.add_argument("--vocab_size", type=int, default=151936)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--hash_key", type=int, default=None,
                    help="override hash_key; use it for the 20-key FPR sweep (C4)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer path, needed when the input has `text` but no `token_ids`")
    ap.add_argument("--split", action="store_true",
                    help="also write <out>_wm.jsonl / <out>_human.jsonl split by `source` "
                         "(run_eval wants watermarked and human attacks in separate files)")
    args = ap.parse_args()

    kgw = load_kgw(args.markllm_dir, args.wm_config, args.vocab_size, args.device,
                   args.hash_key)
    gamma = kgw.config.gamma
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # 透传的附加字段：B 组数据用 valid/attack_frac/actual_attack_frac/sim/token_edit_ratio，
    # 我们自己的攻击用 attacker/attack_strength/passes/orig_id
    EXTRA = ("attack_frac", "actual_attack_frac", "attack_strength", "passes",
             "attacker", "orig_id", "valid", "sim", "token_edit_ratio")

    for path in args.files:
        out = args.out or os.path.join(
            os.path.dirname(path) or ".", "flags_" + os.path.basename(path))
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        rows = [r for r in rows if r.get("valid", True)]
        # 断点续跑：同一条文本可能有多个 attack_frac 版本，所以键是 (id, attack_frac)
        done = set()
        if os.path.exists(out):
            done = {(json.loads(line)["id"], json.loads(line).get("attack_frac"))
                    for line in open(out, encoding="utf-8")}
        todo = [r for r in rows if (r["id"], r.get("attack_frac")) not in done]
        print(f"{path}: {len(rows)} texts, {len(todo)} to do -> {out}")

        with open(out, "a", encoding="utf-8") as fh:
            for s in range(0, len(todo), args.batch):
                chunk = todo[s:s + args.batch]
                seqs = [seq_of(r, tok) for r in chunk]
                fl = flags_for(kgw, seqs, args.vocab_size, args.device)
                fo = first_occurrence(seqs)
                for r, f, o in zip(chunk, fl, fo):
                    rec = {"id": r["id"], "source": r.get("source", ""), "flags": f,
                           "first_occurrence": o}
                    for key in EXTRA:
                        if key in r:
                            rec[key] = r[key]
                    fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"  {min(s + args.batch, len(todo))}/{len(todo)}")

        # sanity check straight from the plan: green ratio ~gamma / ~0.5-0.65, and a z score
        recs = [json.loads(line) for line in open(out, encoding="utf-8")]

        # 按 source 拆成水印文本 / 人写文本两份（攻击对照必须分开统计）
        if args.split:
            for src in ("wm", "human"):
                sub = [r for r in recs if r.get("source") == src]
                if not sub:
                    continue
                dst = out[:-6] + f"_{src}.jsonl" if out.endswith(".jsonl") else out + f"_{src}"
                with open(dst, "w", encoding="utf-8") as fh:
                    for r in sub:
                        fh.write(json.dumps(r) + "\n")
                print(f"  split -> {dst} ({len(sub)} texts)")

        g = sum(map(sum, [r["flags"] for r in recs])) / sum(len(r["flags"]) for r in recs)
        kept = [sum(r["first_occurrence"]) for r in recs]
        print(f"  green ratio {g:.3f} (gamma={gamma}); tokens kept after dedup "
              f"{sum(kept)/sum(len(r['flags']) for r in recs):.3f}")
        if recs:
            f0 = recs[0]["flags"]
            z = (sum(f0) - gamma * len(f0)) / (len(f0) * gamma * (1 - gamma)) ** 0.5
            print(f"  first text: n={len(f0)} green={sum(f0)} z={z:.2f} "
                  f"(compare with generate.py's sanity check for the same id={recs[0]['id']})")


if __name__ == "__main__":
    main()
