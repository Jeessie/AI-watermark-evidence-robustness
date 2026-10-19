"""
B6, stand-in for DIPPER: paraphrase attack through an OpenAI-compatible API (default) or ollama.

DIPPER (Krishna et al. 2023) is a 11B paraphrase model; we do not have it, so the attack is a
prompted rewrite. What the attack is *for* is unchanged: destroy the token-level green/red
pattern while keeping the meaning, at a controlled strength, so C can measure the residual
per-token signal.

B group ships the main attack arm (B/attack.py, 200 texts per level, `data/attacked_wm.jsonl`).
This script is the *strong* arm: same protocol, a harsher attacker, used to push the
"attack strength" axis past what a 1.5B self-paraphrase does.

Three knobs:

  --fracs 0.25 0.5 1.0   fraction of sentences rewritten (nested: frac=0.25 subset of 0.5)
  --strengths light strong   light = synonym-level rewrite, strong = rephrase from scratch
  --passes 1 2           passes = rewrite the rewritten text again (round trip through the model)

  python c_eprocess/attack_rewrite.py data/wm.jsonl data/human.jsonl \
         --out data/attacked_api.jsonl --limit 20 --fracs 0.25 0.5 1.0 --strengths light strong

Output keeps the source's prompt_ids (so C1 can still use prompt_ids[-1] as the KGW prefix) and
the attacked `text`; `valid` marks rewrites that came back empty / degenerate.

  # default backend is the dtens API, deepseek-v4.1-flash (needs DTENS_API_KEY); alternatives:
  OPENAI_API_KEY=... python c_eprocess/attack_rewrite.py data/wm.jsonl --out data/attacked_x.jsonl \
         --base_url https://api.openai.com/v1 --model gpt-5-mini --api_key_env OPENAI_API_KEY

  # local ollama (no network, weaker attacker -- 1.5B is the weak end of the axis)
  python c_eprocess/attack_rewrite.py data/wm.jsonl --out data/attacked_local.jsonl \
         --backend ollama --model qwen2.5:1.5b --limit 20

Do not mix attackers in one output file: the file name is what run_eval reports, so use
one file per attacker and let `file` in attack.csv carry the comparison.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
BACKEND = "ollama"          # set from --backend in main()
BASE_URL = "https://api.openai.com/v1"
API_KEY = None
REASONING_EFFORT = "none"

# light ~ DIPPER order 20, strong ~ order 60
PROMPTS = {
    "light": ("Rewrite the following sentence with different words, keeping the meaning exactly. "
              "Keep it one sentence. Output only the rewritten sentence, no quotes, no explanation.\n\n"
              "Sentence: {s}"),
    "strong": ("Rewrite the following sentence from scratch: change the vocabulary and the sentence "
               "structure, but keep the meaning. Output only the new sentence, no quotes, no explanation.\n\n"
               "Sentence: {s}"),
}

SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    parts = [p.strip() for p in SENT.split(text.strip()) if p.strip()]
    return parts


def _post(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def generate(prompt, model, temperature, num_predict, retries=3):
    if BACKEND == "openai":      # any OpenAI-compatible /chat/completions endpoint
        url = BASE_URL.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": num_predict, "stream": False}
        if REASONING_EFFORT:
            # deepseek-v4.1-flash thinks by default; at 256 tokens it never reaches the answer
            # and returns content=null with everything in reasoning_content (1.5 s, 256 tokens).
            # reasoning_effort=none: 0.5 s, 29 tokens, clean sentence. Same text quality here.
            payload["reasoning_effort"] = REASONING_EFFORT
    else:
        url = API
        headers = {"Content-Type": "application/json"}
        payload = {"model": model, "prompt": prompt, "stream": False,
                   "options": {"temperature": temperature, "num_predict": num_predict,
                               "top_p": 0.9}}
    for k in range(retries + 1):
        try:
            resp = _post(url, payload, headers)
            if BACKEND == "openai":
                msg = resp["choices"][0]["message"]
                text = (msg.get("content") or "").strip()
                if not text:
                    # reasoning model burned the budget on reasoning_content: retry
                    raise ValueError("empty content (reasoning model truncated?)")
                return text
            return resp["response"].strip()
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            if k == retries:
                raise
            time.sleep(1.5 * (k + 1))
            print(f"    retry after {e}", file=sys.stderr)


def clean(s):
    s = re.sub(r"<\s*(think|reasoning)\s*>.*?<\s*/\s*(think|reasoning)\s*>", "", s, flags=re.S | re.I)
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(r"^(rewritten sentence|sentence|output)\s*:\s*", "", s, flags=re.I)
    s = " ".join(s.split())
    s = re.split(r"(?<=[.!?])\s", s)[0]          # a chatty model answers with two sentences; keep the first
    if s and s[-1] not in ".!?":
        s += "."
    return s


def ok(orig, new):
    """A rewrite is usable if it is a plausible sentence and actually different.

    Length ratio only, no punctuation requirement: qwen2.5:1.5b forgets the final period often
    and that is not a failed rewrite (clean() adds it back).
    """
    if not new:
        return False
    r = len(new) / max(1, len(orig))
    if not (0.35 <= r <= 3.0):
        return False
    return new.lower() != orig.lower()


def rewrite_sentences(sents, idx, strength, model, temperature, num_predict, pool):
    """Rewrite the sentences at positions `idx`; failed ones keep their original text."""
    jobs = {i: PROMPTS[strength].format(s=sents[i]) for i in idx}
    out = list(sents)
    fails = 0
    futs = {i: pool.submit(generate, p, model, temperature, num_predict) for i, p in jobs.items()}
    for i, f in futs.items():
        new = clean(f.result())
        if ok(sents[i], new):
            out[i] = new
        else:
            fails += 1
    return out, fails


def attack(rec, args, pool, rng):
    sents = split_sentences(rec["text"])
    order = list(range(len(sents)))
    rng.shuffle(order)                      # nested: frac 0.25 ⊂ 0.5 ⊂ 1.0
    rows = []
    for strength in args.strengths:
        for frac in args.fracs:
            k = max(1, round(frac * len(sents)))
            idx = order[:k]
            cur, fails = rewrite_sentences(sents, idx, strength, args.model, args.temperature,
                                           args.num_predict, pool)
            for p in range(2, args.passes + 1):   # extra passes rewrite everything again
                cur, f2 = rewrite_sentences(cur, list(range(len(cur))), strength, args.model,
                                            args.temperature, args.num_predict, pool)
                fails += f2
            text = " ".join(cur)
            rows.append({
                "id": f"{rec['id']}_f{frac}_{strength}_p{args.passes}",
                "orig_id": rec["id"], "source": rec.get("source", ""),
                "prompt_ids": rec.get("prompt_ids", []),
                "attack_frac": frac, "attack_strength": strength, "passes": args.passes,
                "attacker": f"{args.backend}:{args.model}",
                "n_sent": len(sents), "n_rewritten": len(idx),
                "text": text, "valid": fails <= 0.3 * max(1, len(idx)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="jsonl with `text` (data/wm_local.jsonl, data/human.jsonl)")
    ap.add_argument("--out", default="data/attacked_api.jsonl")
    ap.add_argument("--model", default="deepseek-v4.1-flash",
                    help="model id for --backend openai, or any ollama tag")
    ap.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    ap.add_argument("--base_url", default=os.environ.get("DTENS_BASE_URL", "https://api.dtens.com/v1"))
    ap.add_argument("--api_key_env", default="DTENS_API_KEY")
    ap.add_argument("--reasoning_effort", default="none",
                    help='pass this through to reasoning models; "none" turns thinking off '
                         '(otherwise 256 output tokens are all reasoning and content comes back empty). '
                         'Use "" to omit the field.')
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--strengths", nargs="+", default=["light", "strong"], choices=list(PROMPTS))
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--num_predict", type=int, default=256,
                    help="max output tokens; reasoning models burn some of it before the sentence")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N texts per input file")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    global BACKEND, BASE_URL, API_KEY, REASONING_EFFORT
    BACKEND, BASE_URL, REASONING_EFFORT = args.backend, args.base_url, args.reasoning_effort
    API_KEY = os.environ.get(args.api_key_env)
    if args.backend == "openai" and not API_KEY:
        raise SystemExit(f"--backend openai needs the {args.api_key_env} environment variable")
    print(f"attacker: {args.backend}:{args.model}")

    recs = []
    for path in args.files:
        rs = [json.loads(line) for line in open(path, encoding="utf-8")]
        rs = [r for r in rs if r.get("text")]
        if args.limit:
            rs = rs[: args.limit]
        print(f"{path}: {len(rs)} texts")
        recs += rs

    done = set()
    if os.path.exists(args.out):                       # resumable
        done = {json.loads(line)["id"] for line in open(args.out, encoding="utf-8")}
        print(f"{args.out}: {len(done)} already done")

    rng = random.Random(args.seed)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            open(args.out, "a", encoding="utf-8") as fh:
        for n, rec in enumerate(recs, 1):
            rows = attack(rec, args, pool, rng)
            rows = [r for r in rows if r["id"] not in done]
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            nv = sum(r["valid"] for r in rows)
            print(f"  {n}/{len(recs)} {rec['id']}: {len(rows)} variants ({nv} valid) "
                  f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
