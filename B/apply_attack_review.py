"""Apply manual rejection decisions, preserving automatic quality results."""
import argparse
import json
import os
from pathlib import Path

B_DIR = Path(__file__).resolve().parent
ROOT = B_DIR.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('paths', nargs='*', default=[str(ROOT / 'data/attacked_wm.jsonl')])
    ap.add_argument('--decisions', default=str(ROOT / 'configs/attack_manual_review.json'))
    args = ap.parse_args()
    decisions = {(r['source'], r['id'], r['attack_frac']): r for r in json.loads(Path(args.decisions).read_text())}
    for path in args.paths:
        path = Path(path)
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rejected = 0
        for row in rows:
            decision = decisions.get((row['source'], row['id'], row['attack_frac']))
            if decision is not None:
                row.setdefault('automatic_valid', row['valid'])
                row['manual_review'] = decision
                row['valid'] = bool(row['automatic_valid'] and decision['valid'])
                rejected += not row['valid']
        temp = path.with_suffix('.review.tmp')
        with temp.open('w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        print(f'{path}: {rejected} manually rejected records')


if __name__ == '__main__':
    main()
