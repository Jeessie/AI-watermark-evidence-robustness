"""Randomly sample paraphrases, compare them with originals, and save review decisions. No GPU required."""
import argparse
from itertools import zip_longest
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import textwrap

from attack import sentences

ROOT = Path(__file__).resolve().parent.parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row):
    return row['source'], row['id'], row['attack_frac']


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def display(original, attacked):
    width = max(30, min(80, (shutil.get_terminal_size((140, 24)).columns - 5) // 2))

    def lines(text):
        return [part for line in text.splitlines()
                for part in (textwrap.wrap(line, width=width) or [''])]

    print(f"{'Original':<{width}}  |  Paraphrased")
    print('-' * (width * 2 + 5))
    for left, right in zip_longest(lines(original), lines(attacked), fillvalue=''):
        print(f'{left:<{width}}  |  {right}')
    print('-' * (width * 2 + 5))


def changed_blocks(original, row):
    """Use unchanged spans as anchors; adjacent rewrites may split/merge sentences."""
    spans = sentences(original)
    changed = set(row['paraphrased_indices'])
    if any(i < 0 or i >= len(spans) for i in changed):
        return None
    pattern, blocks = [], []
    index = 0
    while index < len(spans):
        if index not in changed:
            pattern.append(re.escape(spans[index]))
            index += 1
            continue
        start = index
        while index < len(spans) and index in changed:
            index += 1
        blocks.append((start + 1, index, ''.join(spans[start:index])))
        pattern.append('(.+?)')
    matched = re.fullmatch(''.join(pattern), row['text'], re.S)
    if matched is None:
        return None
    return [(start, end, text, rewritten)
            for (start, end, text), rewritten in zip(blocks, matched.groups())]


def display_changes(original, row):
    blocks = changed_blocks(original, row)
    if blocks is None:
        print('Cannot reliably align changed passages. Enter f to compare the full texts.')
        return
    if not blocks:
        print('No sentences were paraphrased.')
    for number, (start, end, before, after) in enumerate(blocks, 1):
        location = str(start) if start == end else f'{start}–{end}'
        print(f'\nChanged passage {number}/{len(blocks)} (original sentence(s) {location})')
        display(before.strip(), after.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['wm', 'human', 'all'], default='all')
    parser.add_argument('--attack-frac', type=float, choices=[0.25, 0.5, 1.0])
    parser.add_argument('--count', type=int, default=1, help='Number of records to review (default: 1)')
    parser.add_argument('--seed', type=int, help='Optional seed for reproducible sampling')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--decisions', type=Path, default=ROOT / 'configs/attack_manual_review.json')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be greater than 0')

    decisions = json.loads(args.decisions.read_text()) if args.decisions.exists() else []
    reviewed = {key(row) for row in decisions}
    candidates = []
    for source in (['wm', 'human'] if args.source == 'all' else [args.source]):
        originals = {row['id']: row for row in read_rows(args.data_dir / f'{source}.jsonl')}
        path = args.data_dir / f'attacked_{source}.jsonl'
        for row in read_rows(path):
            if key(row) in reviewed or 'manual_review' in row:
                continue
            if args.attack_frac is not None and row['attack_frac'] != args.attack_frac:
                continue
            if row['id'] not in originals:
                raise ValueError(f"Original text missing: {source} / {row['id']}")
            candidates.append((path, row, originals[row['id']]['text']))
    if not candidates:
        print('No unreviewed records match the selected filters.')
        return

    selected = random.Random(args.seed).sample(candidates, min(args.count, len(candidates)))
    print('Showing changed passages only; reviewed records are skipped. f: full texts, s: skip, q: quit.')
    print('Adjacent rewrites are grouped to handle sentence splitting. Do not run during attack generation.')
    for number, (path, row, original) in enumerate(selected, 1):
        automatic = row.get('automatic_valid', row['valid'])
        print(f"\n[{number}/{len(selected)}] {row['source']} / {row['id']} / "
              f"Attack fraction: {row['attack_frac']:.0%} / Automatic filter valid={automatic}")
        display_changes(original, row)
        while True:
            answer = input('Does the paraphrase preserve the meaning? valid [true/false/f/s/q]: ').strip().lower()
            if answer == 'f':
                display(original, row['text'])
                continue
            if answer in ('q', 's'):
                break
            if answer in ('true', 'y', 'yes', '1', 'false', 'n', 'no', '0'):
                break
            print('Enter true or false, f for full texts, s to skip, or q to quit.')
        if answer == 'q':
            return
        if answer == 's':
            continue
        valid = answer in ('true', 'y', 'yes', '1')
        while True:
            reason = input('Reason (required if invalid; otherwise press Enter): ').strip()
            if reason or valid:
                break
            print('Please explain how the meaning changed.')
        decision = dict(source=row['source'], id=row['id'], attack_frac=row['attack_frac'],
                        valid=valid, reason=reason or 'No meaning change found during manual review.')
        # Save decisions first so apply_attack_review.py can recover if the second write fails.
        decisions.append(decision)
        save(args.decisions, json.dumps(decisions, ensure_ascii=False, indent=2) + '\n')
        current = read_rows(path)
        for item in current:
            if key(item) == key(row):
                item.setdefault('automatic_valid', item['valid'])
                item['manual_review'] = decision
                item['valid'] = bool(item['automatic_valid'] and valid)
                final_valid = item['valid']
        save(path, ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in current))
        print(f'Saved. Final valid={final_valid}.')
        if valid and not automatic:
            print('Manual review passed, but the automatic quality filter failed; the record remains invalid.')
    print('Review complete. If validity changed, refresh the detection summaries and report.')


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print('\nExited. Previous decisions are saved; the unfinished review was not saved.')
