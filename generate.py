"""
Step A2: generate KGW-watermarked continuations (GPU).

Reads data/prompts.jsonl, writes data/wm.jsonl (or data/nowm.jsonl with --no_watermark).
Each record: id, source, prompt_ids, token_ids (exactly --new_tokens tokens), text.
Resumable: ids already in the output file are skipped, so a Colab disconnect loses at most one batch.
Throughput is appended to logs/throughput.jsonl (needed for the GPU-hour / cost estimate).

Usage (Colab, from the project folder):
  python generate.py --markllm_dir MarkLLM --limit 10      # pilot: 10 texts, check speed
  python generate.py --markllm_dir MarkLLM                 # all prompts
  python generate.py --markllm_dir MarkLLM --no_watermark --limit 50   # optional: unwatermarked model text
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markllm_dir", default="MarkLLM")
    ap.add_argument("--wm_config", default="configs/KGW_g025.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--new_tokens", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_watermark", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    wm_config = os.path.abspath(args.wm_config)
    sys.path.insert(0, os.path.abspath(args.markllm_dir))
    from watermark.auto_watermark import AutoWatermark
    from utils.transformers_config import TransformersConfig

    out_path = args.out or ("data/nowm.jsonl" if args.no_watermark else "data/wm.jsonl")
    source = "nowm" if args.no_watermark else "wm"
    device = args.device
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4 has no bf16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    # vocab_size must match the logits dimension (151936 for Qwen2.5), not len(tokenizer).
    # Detection later must use the same value.
    tc = TransformersConfig(model=model, tokenizer=tok, vocab_size=model.config.vocab_size, device=device)
    kgw = AutoWatermark.load("KGW", algorithm_config=wm_config, transformers_config=tc)
    processors = None if args.no_watermark else LogitsProcessorList([kgw.logits_processor])

    prompts = load_jsonl(args.prompts)
    done = {r["id"] for r in load_jsonl(out_path)}
    todo = [p for p in prompts if p["id"] not in done]
    if args.limit is not None:
        todo = todo[: max(0, args.limit - len(done))]
    print(f"{len(done)} done, {len(todo)} to generate -> {out_path}")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0, n_new = time.time(), 0

    with open(out_path, "a") as fout:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b: b + args.batch_size]
            # Left-pad prompt ids ourselves so the last position is always a real prompt token.
            L = max(len(p["prompt_ids"]) for p in batch)
            input_ids = torch.tensor([[pad_id] * (L - len(p["prompt_ids"])) + p["prompt_ids"] for p in batch], device=device)
            attn = torch.tensor([[0] * (L - len(p["prompt_ids"])) + [1] * len(p["prompt_ids"]) for p in batch], device=device)

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    logits_processor=processors,
                    do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                    repetition_penalty=args.repetition_penalty,
                    min_new_tokens=args.new_tokens, max_new_tokens=args.new_tokens,
                    pad_token_id=pad_id,
                )
            gen = out[:, L:].tolist()
            for p, ids in zip(batch, gen):
                fout.write(json.dumps({
                    "id": p["id"], "source": source,
                    "prompt_ids": p["prompt_ids"], "token_ids": ids,
                    "text": tok.decode(ids, skip_special_tokens=True),
                }) + "\n")
            fout.flush()
            n_new += len(batch) * args.new_tokens
            el = time.time() - t0
            print(f"  {b + len(batch)}/{len(todo)}  {n_new / el:.0f} tok/s  elapsed {el / 60:.1f} min")

    el = time.time() - t0
    if n_new:
        os.makedirs("logs", exist_ok=True)
        with open("logs/throughput.jsonl", "a") as f:
            f.write(json.dumps({
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device, "model": args.model, "source": source,
                "batch_size": args.batch_size, "texts": len(todo), "new_tokens": n_new,
                "seconds": round(el, 1), "tok_per_s": round(n_new / el, 1),
            }) + "\n")

    # Sanity check: watermarked z should be far above 4, human z near 0.
    print("\nsanity check (z-score on the first 3 texts of each file):")
    for path in [out_path, "data/human.jsonl"]:
        for r in load_jsonl(path)[:3]:
            ids = torch.tensor(r["prompt_ids"][-1:] + r["token_ids"], device=device)  # last prompt token seeds the first green list
            z, flags = kgw.utils.score_sequence(ids)
            green = sum(f == 1 for f in flags) / len(r["token_ids"])
            print(f"  {r['source']:6s} {r['id']:12s} z={z:6.2f}  green={green:.2f}")


if __name__ == "__main__":
    main()
