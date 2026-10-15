"""Validate B output contract and summarize quality by source and strength."""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from statistics import mean

B_DIR = Path(__file__).resolve().parent
ROOT = B_DIR.parent
os.environ.setdefault('HF_HOME', str(B_DIR.parent / '.cache/huggingface'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('paths', nargs='*', default=[str(ROOT / 'data/attacked_wm.jsonl'), str(ROOT / 'data/attacked_human.jsonl')])
    ap.add_argument('--verify-tokens', action='store_true')
    args = ap.parse_args()
    tokenizer = None
    if args.verify_tokens:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', local_files_only=True)
    groups = defaultdict(list)
    documents = defaultdict(dict)
    for path in args.paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                for field in ('id', 'source', 'attack_frac', 'text', 'token_ids', 'prompt_ids',
                              'n_sent', 'n_paraphrased', 'sim', 'token_edit_ratio', 'len_ratio', 'valid'):
                    assert field in row, field
                assert row['prompt_ids'] and row['token_ids']
                assert len(row['paraphrased_indices']) == row['n_paraphrased']
                if tokenizer is not None:
                    assert tokenizer(row['text'], add_special_tokens=False)['input_ids'] == row['token_ids'], row['id']
                key = row['source'], row['id']
                frac = row['attack_frac']
                assert frac not in documents[key], f'duplicate {key} {frac}'
                documents[key][frac] = set(row['paraphrased_indices'])
                groups[row['source'], frac].append(row)
    for key, versions in documents.items():
        assert set(versions) == {0.25, 0.5, 1.0}, f'incomplete {key}'
        assert versions[0.25] <= versions[0.5] <= versions[1.0], key
    summary = []
    for (source, frac), rows in sorted(groups.items()):
        summary.append(dict(source=source, attack_frac=frac, n=len(rows),
            valid=sum(r['valid'] for r in rows),
            **{key: mean(r[key] for r in rows) for key in ('sim', 'token_edit_ratio', 'len_ratio', 'actual_attack_frac')}))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
