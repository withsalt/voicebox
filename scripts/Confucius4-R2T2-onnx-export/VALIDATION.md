# Validation record

Test date: 2026-09-20/21. Windows x64, Ryzen 9 9950X, RTX 5070 Ti 16 GB, approximately 64 GB RAM. .NET SDK 10.0.401, ONNX Runtime Python/.NET 1.28.0, PyTorch 2.14.0+cu132, transformers 4.57.6, qwen-asr 0.0.6. CUDA 13 / cuDNN 9 DLLs were supplied from the isolated environment for local testing; deployment does not require Python.

## Independent correctness baseline

The original PyTorch encoder and decoder were compared against the export wrappers. The original eager encoder receives a window mask adapter because qwen-asr 0.0.6 does not apply `cu_seqlens` in eager attention; the adapter restores the window boundaries used by its vLLM/FlashAttention implementation. We did not run vLLM on this Windows host.

Encoder inputs: 80, 160, 320, 990, 1000, 1010, 6740, 8010 and 16130 ms. Wrapper/PyTorch maximum error was below 6.4e-8. The first eight decoder steps passed logits tolerance and greedy-token comparisons. C# validates Python-generated Mel features, arbitrary sample-count boundaries, prompt IDs and multilingual/Unicode tokenizer fixtures.

On the supplied 6.74 s example, FP32 ONNX encoder maximum error was 1.21e-6 and initial logits error 4.01e-5. All three final precisions produced the same offline and streaming example transcriptions as the original reference. FP16 had encoder error 0.00155 and initial logits error 0.110. Final eight-bit weight quantization had initial logits error below 0.786. These are per-example observations, not universal numerical bounds.

Offline and streaming have separate reference transcripts. On the example, streaming omits a comma that offline decoding includes; the original PyTorch streaming policy produces the same difference.

## Chinese offline evaluation

189 existing labeled speech files, 1113.647 s total, 3990 normalized reference characters. Includes AIShell-1, AIShell-3 conversation/test, THCHS clean/noisy and Wenet subsets. Audio files and labels were checked for presence; labels were not rewritten to fit predictions.

| Runtime | Provider | CER | Aggregate RTF | Exact text vs PyTorch | Normalized text vs PyTorch |
|---|---|---:|---:|---:|---:|
| Original PyTorch FP32 | CUDA | 2.456% | 0.0736 | 189/189 | 189/189 |
| C# ONNX FP32 | CUDA | 2.456% | 0.0511 | 189/189 | 189/189 |
| C# ONNX FP16 | CUDA | 2.456% | 0.0441 | 188/189 | 189/189 |
| C# ONNX INT8 weights / FP32 activations | CUDA | 2.481% | 0.0391 | 183/189 | 188/189 |

RTF uses total processing time divided by the actual, unique input duration. The FP16 run briefly overlapped the separate INT8 example validation at startup. Main inference benchmarks run sequentially; lightweight development checks ran during parts of the task. These are local descriptive timings, not isolated hardware comparisons or statistically established speedups between precisions.

## English offline evaluation

73 LibriSpeech dummy test-clean fixtures, 1169 normalized reference words. Source: `hf-internal-testing/librispeech_asr_dummy`, clean validation parquet. This small subset is useful for regression and is not a representative English benchmark.

| Runtime | Provider | WER | Aggregate RTF | Exact text vs PyTorch |
|---|---|---:|---:|---:|
| Original PyTorch FP32 | CUDA | 4.619% | 0.0795 | 73/73 |
| C# ONNX FP16 | CUDA | 4.619% | 0.0448 | 73/73 |
| C# ONNX INT8 weights / FP32 activations | CUDA | 4.619% | 0.0434 | 71/73 |

INT8 normalized English text matched the original reference on 73/73 samples. Two English streaming clips were additionally run through original PyTorch with the same policy and matched C# FP16 exactly.

The final INT8 format uses eight-bit blockwise weights and FP32 activations (`MatMulNBits`, block size 32, accuracy level 1), quantizing 1 encoder MatMul and 197 decoder MatMuls while preserving encoder Gemm/convolution in FP32. CUDA kernel assignment was explicitly profiled. This is a mixed-precision weight-compression option. CPU executes it correctly but is slow: the 6.74 s example took approximately 36 s in Python and 37.86 s in the C# offline demo (RTF 5.62). It is not a CPU realtime recommendation. An earlier dynamic-activation INT8 candidate was rejected after Chinese CER rose to 3.734% and Python/C# streaming text diverged.

## Real-time feeding

FP16 CUDA, 160 ms inference updates, 160 ms initial lookahead, one-token rollback, 20 ms input callbacks. Six Chinese/English samples range from 2.595 to 29.4 s. Completed windows are finalized every 16 s.

- Inference RTF: 0.303–0.343.
- Per-call P95: 67.2–73.6 ms; the largest individual call among these cases was 110.8 ms.
- Finish/drain time: 35.6–73.3 ms.
- First committed text: 1.183–1.503 s after playback starts. This includes leading audio and the model's evidence/commit delay; it is not equivalent to per-call compute time or word-aligned recognition latency.
- No input overflow or committed-prefix retraction occurred in these cases.

The dataset does not provide word alignments for these runs; a word-aligned 200–600 ms latency claim has not been established.

The continuous test concatenated the supplied clip and one second of silence 24 times: 185.76 s, 1150 inference calls. It completed in 185.858 s with inference RTF 0.294, P50 47.3 ms, P95 65.6 ms, maximum 92.1 ms and 83.7 ms final drain. First committed text was at 1.831 s. No queue overflow or committed-prefix retraction occurred. Normalized CER against the repeated example transcript was 2.083%; hard window boundaries caused some word losses. Use VAD utterance boundaries for conversation.

Memory sampling observed process private bytes rising from 7.53 GB initially to 10.14 GB after the first full window, then 10.46 GB at the last sample; working set rose from 1.62 to 2.17 GB. Managed memory fluctuated with allocation/GC. This records actual allocator warmup and growth; a three-minute run is not proof of leak freedom. GPU memory observed during the run was approximately 9.36 GB including other desktop processes.

The final INT8 CUDA package also completed the same six realtime clips: inference RTF 0.341–0.478, per-call P95 69.4–103.8 ms, maximum individual inference 116.6 ms, final drain 57.7–109.3 ms and first committed text 1.190–1.508 s. No input overflow or committed-prefix retraction occurred. FP16 remains the default recommendation for stream latency. The INT8 package occupies approximately 4.45 GB, FP16 4.72 GB and FP32 9.42 GB; these include external weights and runtime assets, not runtime working memory.

## Review fixes

The implementation review and executable comparisons found and corrected:

1. Extra final STFT frame in the shared C# Whisper feature extractor.
2. Short-chunk convolution padding that would change 80–990 ms inputs.
3. Missing acoustic window boundaries in an unmodified eager reference / naive export.
4. External-data offsets in quantized Constant attributes, and small shape constants that must remain inline.
5. TF32 rounding in FP32 validation; FP16 keeps Tensor Core plans enabled.
6. Finalization at exact chunk boundaries, bounded queues with explicit overflow, native-resource disposal ordering and concurrent inference isolation.
7. Unicode-safe BPE rollback, immutable session state despite mutable published result objects, and explicit errors on final token-budget exhaustion.

Deployment packages contain the original license files, complete tokenizer assets, exact Mel filters, graph/weight file sizes and SHA-256 digests. Each final graph passes ONNX checker and an actual ORT session load before its output directory is published. FP32/FP16 were regenerated with the final packaging logic and rechecked against the independent reference.

Final delivery verification independently recomputed all 17 asset hashes in each of `outputs/fp32`, `outputs/fp16` and `outputs/int8`; all matched their manifests. Older trial/export directories remain locally because automatic approval review blocked their cleanup. Only the three directories named above are final deployment packages.

Final automated checks: 6 Python tests; 22 C# tests passed for each final precision on CUDA, including real-model offline/streaming parity; 3 existing Qwen3 lifecycle tests passed. The WAV streaming demo was executed successfully. The microphone code compiled, but an interactive microphone recording was not performed during this task. Package tests cover schema/path validation, truncated assets and checksum mismatch. Dynamic encoder export tests cover 1–1601 Mel frames against independently executed original model layers.

Code review and tests reduce risk but do not prove absence of all bugs. Other GPUs, Linux/native-library combinations, many simultaneous streams and languages beyond Chinese/English require their own acceptance tests.

Raw local reports live under `reports/`, excluded from Git. Reference fixtures can be regenerated using the commands in README. Exported weights remain governed by the bundled model license.
