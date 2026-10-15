"""
Recompute KGW z-scores for whole files (no model needed). Used to check that detection gives the same
result on every machine.

  python verify_kgw.py --device cpu  data/wm.jsonl data/human.jsonl
  python verify_kgw.py --device cuda data/wm.jsonl data/human.jsonl
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys

import torch

from kgw_cpu import patch_kgw_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--markllm_dir", default="MarkLLM")
    ap.add_argument("--wm_config", default="configs/KGW_g025.json")
    ap.add_argument("--vocab_size", type=int, default=151936)  # Qwen2.5 logits dimension
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    wm_config = os.path.abspath(args.wm_config)
    sys.path.insert(0, os.path.abspath(args.markllm_dir))
    from watermark.kgw.kgw import KGW, KGWConfig
    from utils.transformers_config import TransformersConfig

    tc = TransformersConfig(model=None, tokenizer=None, vocab_size=args.vocab_size, device=args.device)
    kgw = patch_kgw_cpu(KGW(KGWConfig(wm_config, tc)))
    g = kgw.config.gamma
    dev = "cuda " + torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    print(f"torch {torch.__version__} on {dev}")

    for path in args.files:
        rs = [json.loads(l) for l in open(path)]
        seqs = [r["prompt_ids"][-1:] + r["token_ids"] for r in rs]  # last prompt token seeds the first green list
        by_prev = collections.defaultdict(list)
        for k, s in enumerate(seqs):
            for i in range(1, len(s)):
                by_prev[s[i - 1]].append((k, i, s[i]))
        flags = [[0] * (len(s) - 1) for s in seqs]
        mask = torch.zeros(args.vocab_size, dtype=torch.bool, device=args.device)
        for prev, items in by_prev.items():
            mask.zero_()
            mask[kgw.utils.get_greenlist_ids(torch.tensor([prev], device=args.device))] = True
            hits = mask[torch.tensor([c for _, _, c in items], device=args.device)].tolist()
            for (k, i, _), h in zip(items, hits):
                flags[k][i - 1] = int(h)

        zs = []
        for f in flags:
            n, G = len(f), sum(f)
            zs.append((G - g * n) / math.sqrt(n * g * (1 - g)))
        s = sorted(zs)
        digest = hashlib.md5(json.dumps(flags).encode()).hexdigest()[:10]
        print(f"{path}: n={len(rs)} green={sum(map(sum, flags)) / sum(map(len, flags)):.3f} "
              f"z min/median/max={s[0]:.2f}/{s[len(s) // 2]:.2f}/{s[-1]:.2f} "
              f"z>4: {sum(z > 4 for z in zs)}  z>2.33: {sum(z > 2.33 for z in zs)}  flags_md5={digest}")


if __name__ == "__main__":
    main()
