"""B9: score original and attacked wm with the local MarkLLM KGW implementation."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

B_DIR = Path(__file__).resolve().parent
ROOT = B_DIR.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--markllm-dir', default=str(ROOT / 'MarkLLM'))
    ap.add_argument('--original', default=str(ROOT / 'data/wm.jsonl'))
    ap.add_argument('--attacked', default=str(ROOT / 'data/attacked_wm.jsonl'))
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    import torch
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(args.markllm_dir).resolve()))
    from watermark.kgw.kgw import KGW, KGWConfig
    from utils.transformers_config import TransformersConfig
    from kgw_cpu import patch_kgw_cpu
    tc = TransformersConfig(model=None, tokenizer=None, vocab_size=151936, device=args.device)
    kgw = patch_kgw_cpu(KGW(KGWConfig(str(ROOT / 'configs/KGW_g025.json'), tc)))
    originals = [json.loads(l) for l in open(args.original)][:args.limit]
    chosen = {r['id'] for r in originals}
    attacked = [json.loads(l) for l in open(args.attacked) if json.loads(l)['id'] in chosen]
    groups = {}
    rows = []
    for row in originals + attacked:
        ids = row['prompt_ids'][-1:] + row['token_ids']
        z, flags = kgw.utils.score_sequence(torch.tensor(ids, device=args.device))
        frac = row.get('attack_frac', 0)
        groups.setdefault(frac, []).append(z)
        rows.append(dict(id=row['id'], attack_frac=frac, z=z, valid=row.get('valid', True),
                         green_ratio=sum(flags[1:])/len(flags[1:])))
    baseline_ok = all(z > 4 for z in groups[0])
    summary = []
    for frac in sorted(groups):
        selected = [r for r in rows if r['attack_frac'] == frac and r['valid']]
        summary.append(dict(attack_frac=frac, n_valid=len(selected),
            mean_z=mean(r['z'] for r in selected) if selected else None,
            mean_green_ratio=mean(r['green_ratio'] for r in selected) if selected else None,
            detected_z4=sum(r['z'] > 4 for r in selected)))
    print(json.dumps(dict(baseline_ok=baseline_ok, torch_version=torch.__version__,
        device=args.device, warning=None if baseline_ok else
        'Original wm baseline not detected. Do not infer attack effectiveness; match generation implementation and RNG first.',
        records=rows, summary_valid=summary,
        mean_z={str(k): mean(v) for k,v in sorted(groups.items())}), indent=2))
    if not baseline_ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
