import copy
import unittest

from compare_benchmarks import compare


def report():
    return {'provider': 'cuda', 'mode': 'realtime', 'warmup_iterations': 1, 'cer': 0,
            'cases': [
                {'id': 'short', 'iteration': 0, 'reference': 'a', 'text': 'a', 'audio_seconds': 1,
                 'inference_seconds': .006, 'inference_calls': 3, 'inference_ms': [1, 2, 3],
                 'inference_allocated_bytes': 200, 'finish_ms': 4},
                {'id': 'long', 'iteration': 0, 'reference': 'b', 'text': 'b', 'audio_seconds': 3,
                 'inference_seconds': .2, 'inference_calls': 1, 'inference_ms': [200],
                 'inference_allocated_bytes': 800, 'finish_ms': 6}]}


class BenchmarkComparisonTests(unittest.TestCase):
    def test_percentiles_are_pooled_and_rtf_is_duration_weighted(self):
        before = report()
        after = copy.deepcopy(before)
        for item in after['cases']:
            item['inference_seconds'] /= 2
            item['inference_ms'] = [v / 2 for v in item['inference_ms']]
            item['inference_allocated_bytes'] //= 2
        result = compare(before, after)
        self.assertEqual(2, result['inference_p50_ms']['baseline'])
        self.assertEqual(200, result['inference_p95_ms']['baseline'])
        self.assertAlmostEqual(.206 / 4, result['inference_rtf']['baseline'])
        self.assertEqual(50, result['inference_seconds']['reduction_percent'])
        self.assertEqual(1000, result['inference_allocated_bytes']['baseline'])
        self.assertEqual([], result['text_mismatches'])

    def test_rejects_unpaired_or_differently_warmed_runs(self):
        for mutate in (lambda d: d['cases'].reverse(),
                       lambda d: d.update(warmup_iterations=0),
                       lambda d: d.update(provider='cpu')):
            after = report()
            mutate(after)
            with self.assertRaises(ValueError):
                compare(report(), after)

    def test_text_mismatch_keeps_the_utterance_and_iteration(self):
        after = report()
        after['cases'][0]['text'] = 'changed'
        mismatch = compare(report(), after)['text_mismatches'][0]
        self.assertEqual('short', mismatch['id'])
        self.assertEqual(0, mismatch['iteration'])
        self.assertEqual('changed', mismatch['candidate'])

    def test_old_reports_cannot_supply_pooled_percentiles(self):
        before = report()
        for item in before['cases']:
            del item['inference_ms']
        result = compare(before, report())
        self.assertNotIn('inference_p95_ms', result)
        self.assertIn('percentile_note', result)


if __name__ == '__main__':
    unittest.main()
