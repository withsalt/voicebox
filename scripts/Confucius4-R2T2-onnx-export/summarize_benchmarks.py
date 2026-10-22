"""Score normalized text against human labels and compare runtimes separately."""
import argparse
import json
from pathlib import Path
import unicodedata


def normalized(text, words=False):
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c
                   for c in unicodedata.normalize('NFC', text).lower())
    return text.split() if words else list(''.join(text.split()))


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, nargs='+', required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--english', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    baseline = {} if args.baseline is None else {x['id']: x['text'] for x in json.loads(args.baseline.read_text('utf-8'))['cases']}
    summary = []
    for path in args.reports:
        report = json.loads(path.read_text('utf-8'))
        errors = count = same = normalized_same = 0
        differences = []
        for item in report['cases']:
            ref, hyp = normalized(item['reference'], args.english), normalized(item['text'], args.english)
            errors += distance(ref, hyp); count += len(ref)
            if item['id'] in baseline:
                base = baseline[item['id']]
                same += base == item['text']
                normalized_same += normalized(base, args.english) == hyp
                if base != item['text']:
                    differences.append({'id': item['id'], 'baseline': base, 'actual': item['text']})
        summary.append({'report': path.name, 'count': len(report['cases']),
                        'metric': 'WER' if args.english else 'CER', 'error_rate': errors / max(1, count),
                        'errors': errors, 'reference_units': count, 'rtf': report.get('rtf'),
                        'baseline_exact': same, 'baseline_normalized_exact': normalized_same,
                        'differences': differences})
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in summary:
        print({k: v for k, v in row.items() if k != 'differences'})


if __name__ == '__main__':
    main()
