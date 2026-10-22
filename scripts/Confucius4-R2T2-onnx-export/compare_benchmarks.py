"""Compare paired C# ASR runs. Percentiles use raw calls, never averages of percentiles."""
import argparse
import json
import math
from pathlib import Path
import statistics


def metric(before, after):
    return {'baseline': before, 'candidate': after,
            'reduction_percent': 100 * (1 - after / before) if before else None}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def compare(before, after):
    if before['mode'] != after['mode']:
        raise ValueError('Modes must match')
    if before['mode'] == 'frontend':
        keys = lambda data: [(x['operation'], x['id']) for x in data['results']]
        if keys(before) != keys(after):
            raise ValueError('Frontend cases/order must match')
        return {'mode': 'frontend', 'cases': [
            {'operation': a['operation'], 'id': a['id'],
             'median_ms': metric(statistics.median(a['milliseconds']), statistics.median(b['milliseconds'])),
             'median_allocated_bytes': metric(statistics.median(a['allocated_bytes']), statistics.median(b['allocated_bytes']))}
            for a, b in zip(before['results'], after['results'])]}

    for key in ('provider', 'warmup_iterations'):
        if before.get(key, 0) != after.get(key, 0):
            raise ValueError(f'{key} must match')
    a, b = before['cases'], after['cases']
    keys = lambda cases: [(x['id'], x['iteration'], x['reference'], x['audio_seconds']) for x in cases]
    if not a or keys(a) != keys(b):
        raise ValueError('Cases, iterations, references, durations and order must match')
    a_seconds, b_seconds = (sum(x['inference_seconds'] for x in cases) for cases in (a, b))
    duration = sum(x['audio_seconds'] for x in a)
    result = {
        'mode': before['mode'], 'provider': before['provider'], 'cases': len(a),
        'warmup_iterations': before.get('warmup_iterations', 0), 'audio_seconds': duration,
        'inference_seconds': metric(a_seconds, b_seconds),
        'inference_rtf': metric(a_seconds / duration, b_seconds / duration),
        'inference_calls': metric(sum(x['inference_calls'] for x in a), sum(x['inference_calls'] for x in b)),
        'text_mismatches': [{'id': x['id'], 'iteration': x['iteration'], 'baseline': x['text'], 'candidate': y['text']}
                            for x, y in zip(a, b) if x['text'] != y['text']],
        'cer': {'baseline': before['cer'], 'candidate': after['cer']},
        'finish_median_ms': metric(statistics.median(x['finish_ms'] for x in a), statistics.median(x['finish_ms'] for x in b)),
    }
    if all('inference_ms' in x for x in a + b):
        a_times, b_times = ([v for x in cases for v in x['inference_ms']] for cases in (a, b))
        result['inference_p50_ms'] = metric(percentile(a_times, .5), percentile(b_times, .5))
        result['inference_p95_ms'] = metric(percentile(a_times, .95), percentile(b_times, .95))
    else:
        result['percentile_note'] = 'Raw per-call times missing; pooled P50/P95 cannot be computed.'
    if all('inference_allocated_bytes' in x for x in a + b):
        result['inference_allocated_bytes'] = metric(sum(x['inference_allocated_bytes'] for x in a),
                                                    sum(x['inference_allocated_bytes'] for x in b))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text('utf-8-sig')), json.loads(args.after.read_text('utf-8-sig')))
    result['inputs'] = {'baseline': str(args.before.resolve()), 'candidate': str(args.after.resolve())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
