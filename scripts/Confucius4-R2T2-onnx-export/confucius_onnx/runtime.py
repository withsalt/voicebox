"""Python ORT reference for the portable package (greedy, batch size one)."""
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from .common import prepare, encoder_inputs


class Runtime:
    def __init__(self, directory, provider='cpu'):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text('utf-8'))
        self.dtype = np.float16 if self.metadata['dtype'] == 'float16' else np.float32
        self.gpu = provider == 'cuda'
        if (self.gpu and self.metadata['dtype'] == 'int8' and
                self.metadata.get('quantization', {}).get('method') != 'weight_only_blockwise'):
            raise ValueError('Dynamic INT8 is a CPU package')
        if self.gpu:
            ort.preload_dlls()
        providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC',
                     'arena_extend_strategy': 'kSameAsRequested',
                     'use_tf32': '1' if self.dtype == np.float16 else '0'})] if self.gpu else ['CPUExecutionProvider']
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.enable_mem_pattern = not self.gpu
        self.encoder = ort.InferenceSession(str(self.directory / 'encoder.onnx'), options, providers=providers)
        self.decoder = ort.InferenceSession(str(self.directory / 'decoder.onnx'), options, providers=providers)
        if self.gpu and ('CUDAExecutionProvider' not in self.encoder.get_providers() or
                         'CUDAExecutionProvider' not in self.decoder.get_providers()):
            raise RuntimeError('CUDA was requested but ORT fell back to CPU')
        self.processor = Qwen3ASRProcessor.from_pretrained(str(directory), local_files_only=True)
        self.past_names = [x.name for x in self.decoder.get_inputs()][3:]
        shape = (1, self.metadata['num_key_value_heads'], self.metadata['static_cache_len'], self.metadata['head_dim'])
        self.zeros = [ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, self.dtype), 'cuda' if self.gpu else 'cpu', 0)
                      for _ in self.past_names]

    def encode(self, inputs):
        padded, lengths = encoder_inputs(inputs)
        return self.encoder.run(None, {'padded_feature': padded.numpy().astype(self.dtype),
                                      'chunk_lengths': lengths.numpy()})[0]

    def decode_step(self, ids, features, positions, cache):
        binding = self.decoder.io_binding()
        binding.bind_cpu_input('input_ids', ids)
        binding.bind_cpu_input('audio_features', features)
        binding.bind_cpu_input('cache_position', positions)
        for name, value in zip(self.past_names, cache):
            binding.bind_ortvalue_input(name, value)
        binding.bind_output('logits', 'cpu')
        for name in self.past_names:
            binding.bind_output(name.replace('past_', 'present_'), 'cuda' if self.gpu else 'cpu')
        self.decoder.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        return outputs[0].numpy(), outputs[1:]

    def transcribe(self, audio, language='Chinese', prefix='', max_tokens=256, return_debug=False):
        start = time.perf_counter()
        inputs = prepare(self.processor, audio, language, prefix=prefix)
        features = self.encode(inputs)
        ids = inputs['input_ids'].numpy()
        if ids.shape[1] + max_tokens > self.metadata['static_cache_len']:
            raise ValueError('Prompt and generation budget exceed cache capacity')
        positions = np.arange(ids.shape[1], dtype=np.int64)[None, :]
        cache = self.zeros
        tokens = []
        first_logits = None
        for _ in range(max_tokens):
            logits, cache = self.decode_step(ids, features, positions, cache)
            if first_logits is None:
                first_logits = logits.copy()
            token = int(logits.argmax(-1)[0])
            if token in self.metadata['eos_token_ids']:
                break
            tokens.append(token)
            ids = np.array([[token]], dtype=np.int64)
            positions = positions[:, -1:] + 1
            features = np.zeros((1, self.metadata['audio_output_dim']), self.dtype)
        result = {'text': prefix + self.processor.tokenizer.decode(tokens, skip_special_tokens=True),
                  'tokens': tokens, 'seconds': time.perf_counter() - start}
        if return_debug:
            result['first_logits'] = first_logits
        return result
