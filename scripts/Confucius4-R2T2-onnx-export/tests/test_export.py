from types import SimpleNamespace
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
from confucius_onnx.common import reference_window_attention
from confucius_onnx.graphs import Encoder, TensorCache
from export_confucius4_onnx import consolidate, export_graph, quantize


def test_external_constant_attributes_survive_repacking(tmp_path):
    source = tmp_path / 'source'
    target = tmp_path / 'target'
    source.mkdir(); target.mkdir()
    weight = numpy_helper.from_array(np.arange(4, dtype=np.float32).reshape(2, 2), 'weight')
    constant = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), 'constant')
    graph = helper.make_graph([helper.make_node('Constant', [], ['c'], value=constant),
                               helper.make_node('Add', ['weight', 'c'], ['out'])], 'external', [],
                              [helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 2])], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 20)], ir_version=10)
    onnx.save_model(model, source / 'test.onnx', save_as_external_data=True,
                    all_tensors_to_one_file=True, location='weights.data', size_threshold=0, convert_attribute=True)
    consolidate(source / 'test.onnx', target / 'test.onnx')
    session = ort.InferenceSession(str(target / 'test.onnx'), providers=['CPUExecutionProvider'])
    np.testing.assert_array_equal(session.run(None, {})[0], np.arange(4).reshape(2, 2) + 1)


def test_cache_update_is_functional_and_preserves_other_positions():
    original = [torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 8, 4)]
    cache = TensorCache(original)
    keys, values = cache.update(torch.ones(1, 2, 2, 4), torch.full((1, 2, 2, 4), 2.), 0,
                                {'cache_position': torch.tensor([[2, 3]])})
    assert keys.shape == (1, 2, 4, 4)
    assert torch.count_nonzero(original[0]) == 0
    assert torch.all(keys[:, :, :2] == 0)
    assert torch.all(keys[:, :, 2:] == 1)
    assert torch.all(values[:, :, 2:] == 2)


def test_int8_is_eight_bit_weights_with_float_activations(tmp_path):
    weights = np.random.default_rng(11).normal(0, .1, size=(128, 64)).astype(np.float32)
    graph = helper.make_graph([helper.make_node('MatMul', ['x', 'w'], ['y'], name='linear')],
        'weight8', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 128])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 64])], [numpy_helper.from_array(weights, 'w')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 20)], ir_version=10)
    path = tmp_path / 'model.onnx'
    onnx.save(model, path)
    assert quantize(path) == 1
    packed = onnx.load(path, load_external_data=False)
    node = next(n for n in packed.graph.node if n.op_type == 'MatMulNBits')
    assert next(a.i for a in node.attribute if a.name == 'bits') == 8
    assert not any(n.op_type == 'DynamicQuantizeLinear' for n in packed.graph.node)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    x = np.ones((1, 128), np.float32)
    np.testing.assert_allclose(session.run(None, {'x': x})[0], x @ weights, atol=.025, rtol=.03)


def test_dynamic_encoder_matches_original_at_short_and_long_boundaries(tmp_path):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    config = Qwen3ASRAudioEncoderConfig(d_model=16, encoder_attention_heads=2,
        encoder_layers=2, encoder_ffn_dim=32, downsample_hidden_size=4, output_dim=16,
        num_mel_bins=128, n_window=50, n_window_infer=800)
    config._attn_implementation = 'eager'
    tower = Qwen3ASRAudioEncoder(config).eval()
    reference_window_attention(SimpleNamespace(thinker=SimpleNamespace(audio_tower=tower)))
    encoder = Encoder(tower).eval()
    def make_input(length):
        mel = torch.randn(128, length)
        chunks, width = (length + 99) // 100, min(100, length)
        padded = torch.nn.functional.pad(mel, (0, chunks * width - length))
        padded = padded.reshape(128, chunks, width).permute(1, 0, 2).unsqueeze(1)
        lengths = torch.full((chunks,), 100, dtype=torch.int64)
        lengths[-1] = length - (chunks - 1) * 100
        return mel, padded, lengths
    _, padded, lengths = make_input(123)
    export_graph(encoder, (padded, lengths), tmp_path / 'encoder.onnx',
        ['padded_feature', 'chunk_lengths'], ['audio_features'],
        {'padded_feature': {0: 'chunks', 3: 'width'}, 'chunk_lengths': {0: 'chunks'}, 'audio_features': {0: 'tokens'}})
    session = ort.InferenceSession(str(tmp_path / 'encoder.onnx'), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        for length in [1, 8, 16, 32, 99, 100, 101, 799, 800, 801, 1601]:
            mel, padded, lengths = make_input(length)
            reference = tower(mel, feature_lens=torch.tensor([length])).last_hidden_state.numpy()
            actual = session.run(None, {'padded_feature': padded.numpy(), 'chunk_lengths': lengths.numpy()})[0]
            np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-4)
