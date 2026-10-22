from confucius_onnx.streaming import stream, normalize
import numpy as np


class Characters:
    def encode(self, text): return list(text)
    def decode(self, tokens): return ''.join(tokens)


def test_finish_runs_even_on_exact_boundary_and_flushes_tail():
    calls = []
    def infer(audio, prefix, budget):
        calls.append((len(audio), prefix, budget))
        return prefix + ('结束' if budget == 256 else '文字')
    result = stream(np.zeros(5120, np.float32), Characters(), infer)
    assert [x[2] for x in calls] == [4, 256]
    assert result['text'] == '文结束'


def test_normalization_preserves_english_spaces_and_wait_marker():
    assert normalize(' hello， world！|waiting') == ' hello, world!'
    assert normalize('你好, world。') == '你好， world.'
