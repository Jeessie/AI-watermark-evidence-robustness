"""
Step A1: build prompts (for watermarked generation) and human-written negatives from C4.

No GPU needed. Output:
  data/prompts.jsonl  - n_wm docs:    first PROMPT_LEN tokens -> prompt for generation
  data/human.jsonl    - n_human docs: PROMPT_LEN-token context + next CONT_LEN tokens of human text
The two sets come from disjoint documents.

Usage:
  python prepare_data.py --c4_path MarkLLM/dataset/c4/c4.json          # local copy bundled with MarkLLM
  python prepare_data.py --c4_path hf                                   # stream allenai/c4 realnewslike from HuggingFace
"""
import argparse
import json
import os
import random

from transformers import AutoTokenizer


def iter_c4(c4_path):
    if c4_path == "hf":
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "realnewslike", split="validation", streaming=True)
        for i, row in enumerate(ds):
            yield f"c4hf_{i}", row["text"]
    else:
        with open(c4_path) as f:
            for i, line in enumerate(f):
                yield f"c4_{i}", json.loads(line)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4_path", default="MarkLLM/dataset/c4/c4.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompt_len", type=int, default=50)
    ap.add_argument("--cont_len", type=int, default=600)
    ap.add_argument("--n_wm", type=int, default=200)
    ap.add_argument("--n_human", type=int, default=500)
    ap.add_argument("--max_scan", type=int, default=20000, help="stop scanning C4 after this many docs")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    need = args.prompt_len + args.cont_len
    n_total = args.n_wm + args.n_human

    # Keep only documents long enough to give a full prompt + full human continuation.
    pool = []
    for n_seen, (doc_id, text) in enumerate(iter_c4(args.c4_path)):
        if n_seen >= args.max_scan or len(pool) >= 3 * n_total:
            break
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= need:
            pool.append((doc_id, ids[:need]))
    print(f"scanned {n_seen} docs, {len(pool)} have >= {need} tokens")
    if len(pool) < n_total:
        raise SystemExit(f"only {len(pool)} long docs, need {n_total}; lower n_wm/n_human or cont_len")

    random.Random(args.seed).shuffle(pool)
    wm_docs, human_docs = pool[:args.n_wm], pool[args.n_wm:n_total]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "prompts.jsonl"), "w") as f:
        for doc_id, ids in wm_docs:
            p = ids[:args.prompt_len]
            f.write(json.dumps({"id": doc_id, "prompt_ids": p, "prompt_text": tok.decode(p)}) + "\n")

    with open(os.path.join(args.out_dir, "human.jsonl"), "w") as f:
        for doc_id, ids in human_docs:
            p, c = ids[:args.prompt_len], ids[args.prompt_len:]
            f.write(json.dumps({
                "id": doc_id, "source": "human",
                "prompt_ids": p, "token_ids": c, "text": tok.decode(c),
            }) + "\n")

    print(f"wrote {len(wm_docs)} prompts and {len(human_docs)} human texts to {args.out_dir}/")


if __name__ == "__main__":
    main()
