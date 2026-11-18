# VoiceBox.DNR.GRENet

Streaming GRE-Net dereverberation for VoiceBox. The service implements
`IDigitalNoiseReductionService` and accepts arbitrary `float` chunks while the
causal ONNX model advances one 200-sample hop at 16 kHz.

The bundled causal WHAMR model is a static FP32
export of the `g_00515000` checkpoint with four blocks and a 100-frame
attention window. [GRE-Net-Streaming](https://github.com/Audio-WestlakeU/GRE-Net-Streaming)
The model is distributed under the upstream repository's MIT license.
The package includes that license as `LICENSE`.

The export uses external K/V rings, projected convolution histories, and packed
streaming RMS caches. Transformer features stay contiguous between the time
and frequency passes. ONNX Runtime can fuse the per-row RMS normalization,
and each time-GRU layer projects its input and recurrent gates in two Gemms.
The frequency complex FFN folds explicit input padding into its first
convolution and implements the cropped transposed convolution as a regular
convolution with reversed kernels. Frequency attention uses
`com.microsoft.MultiHeadAttention` with the scale folded into the Q weights.

`GRENetOptions.Model` is empty by default. Both CPU and CUDA load `stream.onnx`.
If your configuration explicitly names `stream-cpu.onnx`, clear `Model` or update the path.
Set it to a file path to load a custom ONNX model. Relative paths resolve from
the working directory. Use `Model` for paths previously set through `ModelPath`.

CPU is the default provider and defaults to two CPU intra-op threads with full
graph optimization. Selecting `OnnxExecutionProvider.Cuda` enables CUDA Graph
replay by default. CUDA precision follows the provider settings described below.
Run `Process` on one persistent worker thread. Each streaming cache bank
has its own capture. For low startup latency in live audio, process a few
200-sample hops on that worker during startup, then call `Reset()`.

The CUDA runner loads the NVIDIA driver from `nvcuda.dll` on Windows or
`libcuda.so.1` on Linux. The host application also needs the ONNX Runtime CUDA
execution provider. The 1.32.0 Linux GPU NuGet package ships `linux-x64`
binaries. On Jetson, supply a `linux-arm64` native ONNX Runtime build with CUDA
support compatible with the managed API and the device's JetPack, CUDA and
cuDNN versions. Both architectures use the same `stream.onnx`.

`ThreadsNum` configures the CPU intra-op threads. `CudaTuning` supplies explicit
CUDA provider settings. Its `AdditionalOptions["enable_cuda_graph"] = "1"` enables
CUDA Graph replay, and `AdditionalOptions["use_tf32"] = "0"` selects FP32 math.
Bundled and custom models use the CUDA provider's default precision unless
configured explicitly. `SessionOptionsFactory` provides full control over session creation.

```csharp
manager
    .AddGRENetDNRService(options =>
    {
        options.SampleRate = 16000;
        options.ChannelCount = 1;
        options.SampleFormat = SampleFormat.Float32;
        options.ExecutionProvider = OnnxExecutionProvider.Cuda;
    });
```

### Time convolution and normalization caches

The current export has 1,605 nodes. Each time-transformer convolution stores
its three preceding frames in a separate tensor, ordered by tap, frequency bin
and channel. This lets the model append or discard a complete frame in one
operation. The projection reads the four taps in the checkpoint's original
channel and tap order. The encoder and decoder each sum their normalization
histories in one operation over the time axis, preserving the individual
statistics for every channel and layer.

The streaming signature has 35 inputs and 36 outputs. Standalone callers can
allocate zero-filled caches from the input metadata. The eight
`ts_ang_{in,mid}_history_{1..4}` tensors replace the four real/imaginary FFN
banks in the 31-input model. Feed each `*_out` cache directly into its matching
input on the next hop. For each of the eight K/V `*_frame` outputs, write the
frame to `ring[:, cursor, :]` using that hop's input cursor. VoiceBox handles
both signatures automatically.

On both upstream recordings (131.83 seconds, 2,109,251 samples),
Release `GRENetDNRService.ProcessAll` with 137-sample chunks produced
bit-identical float32 waveforms to the bundled model from `2b76f77`.
This held separately on CPU with two threads, CUDA with provider defaults,
and CUDA with TF32 disabled.

On Windows with a Ryzen 9 9950X, an RTX 5070 Ti and ONNX Runtime 1.32.0,
the Release `benchmark.cs` harness timed `GRENetDNRService.Process` using
the same input and provider settings for each pair. Each graph ran three
times per configuration, reversing the order on the second round. Each run
used 120 warmup hops and 10 seconds of timed audio. The table gives the
medians of the three per-run means and P99 values.

| Provider | Samples per call | Mean baseline → current | Mean reduction | P99 baseline → current |
|---|---:|---:|---:|---:|
| CPU, 2 threads | 200 | 5.508 → 5.056 ms | 8.2% | 5.874 → 5.422 ms |
| CPU, 2 threads | 256 | 7.089 → 6.470 ms | 8.7% | 11.727 → 10.467 ms |
| CUDA Graph, TF32 on | 200 | 3.673 → 3.588 ms | 2.3% | 4.621 → 4.538 ms |
| CUDA Graph, TF32 off | 200 | 4.493 → 4.410 ms | 1.9% | 5.555 → 5.431 ms |

### Previous 31-input FP32 export

The 31-input graph had 1,751 nodes, down from 2,022 in the bundled baseline at
VoiceBox commit `4219060`. The benchmark measures Release
`GRENetDNRService.Process` on Windows 11 with ONNX Runtime 1.32.0, a Ryzen 9
9950X and an RTX 5070 Ti. The baseline and new export ran in alternating order
with three runs per model, 120 warmup hops and 5 seconds of audio per run.
Mean and P99 are the medians of their respective per-run statistics.

| Provider | Samples per call | Mean baseline → current | P99 baseline → current | RTF baseline → current |
|---|---:|---:|---:|---:|
| CPU, 2 threads | 200 | 6.14 → 5.30 ms | 6.57 → 5.58 ms | 0.491 → 0.424 |
| CUDA Graph, TF32 off | 200 | 4.80 → 4.44 ms | 5.77 → 5.50 ms | 0.384 → 0.355 |
| CPU, 2 threads | 256 | 7.84 → 6.77 ms | 12.74 → 10.93 ms | 0.490 → 0.423 |

All calls finished within their respective 12.5 ms and 16 ms budgets in these
throughput runs. The CUDA row uses full FP32 math for both models. The previous
bundled defaults used TF32 and measured a 4.07 ms mean on the same hardware in
a separate run series.

### External K/V ring export on low-power CPU

On an Ubuntu Intel Core 3 100U, the default CPU runner keeps the 99 previous
K/V slots in persistent buffers. Each hop returns eight small `*_frame` tensors.
After inference the runner copies their rows into the current slot and advances
the ring. The graph scores those past frames together with the current frame.
This reduces K/V writeback from about 31 MB to 0.31 MB per hop. The dense
convolutions also cache the result of the delayed kernel tap, which uses 80
channels at each depth. Both transformations preserve the 100-frame attention
window and the checkpoint's FP32 weights.

The following numbers are from Release `GRENetDNRService.Process` with ORT
1.32.0, CPU full graph optimization, 16 kHz mono, and 200 samples per call.
The runs used normal Linux scheduling and the same deterministic input. Each
configuration ran three times with 120 warmup hops and 800 timed hops per run.
Mean and P99 are the medians of the three per-run statistics. The rightmost
column counts misses across all 2,400 calls.

| CPU graph | Threads | Mean | P99 | RTF | Over 12.5 ms |
|---|---:|---:|---:|---:|---:|
| Previous full-state graph | 4 | 15.59 ms | 16.09 ms | 1.247 | 2400/2400 |
| Previous full-state graph | 2 | 15.00 ms | 15.15 ms | 1.200 | 2400/2400 |
| External K/V ring graph | 2 | 11.30 ms | 11.72 ms | 0.904 | 5/2400 |

A separate run using the bundled defaults and 12.5 ms scheduled arrivals
finished 4 of 800 calls after their deadlines. With 256-sample callbacks some
calls execute two hops. One 625-call throughput run had a P99 of 22.71 ms and
175 calls exceeded the 16 ms callback period.

On CUDA the runner writes each frame into the persistent GPU rings with
strided device-to-device copies. Both runners also accept older full-state
graphs supplied through `Model`.

### External K/V ring export on CUDA

Windows x64, RTX 5070 Ti, ONNX Runtime 1.32.0 with CUDA Graph replay. The same
Release `Process` benchmark, input and 200-sample callback were used for both
graphs, including cache writes. Each configuration ran three times with 120
warmup and 800 timed hops per run. Mean and P99 are medians across runs.

| CUDA graph | Mean | P99 | Over 12.5 ms |
|---|---:|---:|---:|
| Previous full-state graph | 5.76 ms | 6.71 ms | 0/2400 |
| External K/V ring graph | 4.55 ms | 5.80 ms | 0/2400 |

### Frequency attention fusion (2026-11-18)

The exporter rewrites each of the four non-causal frequency-transformer
attention subgraphs into one `com.microsoft.MultiHeadAttention` node. Q, K and V
enter in their natural `[1, 101, 96]` projection layout and the `[1, 101, 96]`
result feeds the original output projection; the six input-layout nodes, the
merge transpose and the two attention MatMuls plus Softmax disappear. Because
the packed query weight already contains the `head_dim ** -0.5` scale, the fused
node must use `scale=1.0` (using the default scale double-scales the scores).

The graph drops from 2058 to 2022 nodes at opset 17 with the `com.microsoft`
domain. Paired measurements on the same machine:

| Graph | Provider | Mean | P99 | RTF |
|---|---|---:|---:|---:|
| Before fusion | CPU 2 threads | 6.545 ms | 6.99 ms | 0.524 |
| Fused | CPU 2 threads | 6.461 ms | 6.86 ms | 0.517 |
| Before fusion | CUDA | 4.008 ms | 5.07 ms | 0.321 |
| Fused | CUDA | 3.960 ms | 4.58 ms | 0.317 |

That export produced a complex-spectrum maximum error of `2.64e-5` and
113.86 dB against the PyTorch checkpoint; on the recorded-audio set the fused
model is 128.9–134.3 dB against the previous ONNX with identical 16-bit output
except for isolated 1-LSB rounding. `com.microsoft.MultiHeadAttention` is
verified on ONNX Runtime 1.32 CPU and CUDA.

### Node-minimal export (2026-11)

The node-minimal step alone reduces the graph from 2477 to 2058 nodes; the
attention fusion above brings it to 2022. Measured on Windows 11 with a Ryzen 9
9950X and ONNX Runtime 1.32.0, Release build, 120 warmup and 800 timed hops per
run, paired against the previous graph:

| Graph | Provider | Threads | Mean | P99 | RTF |
|---|---|---:|---:|---:|---:|
| Previous | CPU | 2 | 6.99 ms | 7.73 ms | 0.559 |
| Node-minimal | CPU | 2 | 6.50 ms | 6.92 ms | 0.520 |
| Previous | CPU | 1 | 7.68 ms | 8.94 ms | 0.614 |
| Node-minimal | CPU | 1 | 7.14 ms | 7.85 ms | 0.571 |
| Previous | CUDA | 1 | 4.45 ms | 4.79 ms | 0.356 |
| Node-minimal | CUDA | 1 | 3.95 ms | 4.25 ms | 0.316 |

The paired CPU runs were about 7% faster and the CUDA runs about 11% faster.
The attention fusion adds a further ~1% CPU / ~1% CUDA. `Process` time still
depends on background load, so treat single-run CPU comparisons as noisy; the
CUDA numbers were stable to within 0.02 ms.

Run the benchmark from the repository root. The last argument is the callback size.
The benchmark uses `use_tf32=0` on CUDA. Set `GRENET_CUDA_TF32=1` to measure
with TF32 enabled.

```powershell
$model = 'src/Components/VoiceBox.DNR.GRENet/runtimes/models/dnr/grenet/stream.onnx'
$bench = 'scripts/grenet-onnx-export/benchmark.cs'
$env:GRENET_SECONDS = '5'
dotnet run --file $bench -c Release -- $model Cpu 2 200
dotnet run --file $bench -c Release -- $model Cuda 1 200
```

### Reproducing the optimized export

From the VoiceBox repository root, using the upstream GRE-Net-Streaming checkout
at revision `934842363c3366f2934d1cca821d6dcbc4875f21` and the `voicebox`
Python environment, run the following commands. The validator compares 256
consecutive hops with the PyTorch checkpoint and checks the FFN histories at
startup and across two attention-ring wraps. The exporter writes
`stream.onnx` to the current directory by default. Use `--output` to set the
destination path.

```powershell
$source = 'scripts/grenet-onnx-export/GRE-Net-Streaming'
$model = 'src/Components/VoiceBox.DNR.GRENet/runtimes/models/dnr/grenet/stream.onnx'
conda run -n voicebox python -s scripts/grenet-onnx-export/export.py --source $source --output $model
conda run -n voicebox python -s scripts/grenet-onnx-export/validate.py --source $source --model $model
```

The bundled export produced a complex-spectrum maximum absolute error of `1.44e-5`
and a signal-to-error ratio of 114.19 dB relative to the PyTorch checkpoint.
To check bitwise equivalence of the amplitude, phase and complex outputs on
CPU, set `$baseline` to the bundled `stream.onnx` from commit `2b76f77` and run
`conda run -n voicebox python -s scripts/grenet-onnx-export/compare_models.py $baseline $model --threads 2 --require-exact`.

`export.py` applies the frequency-attention fusion (`optimize_attention.py`)
automatically; the result uses the `com.microsoft` domain and requires an ONNX
Runtime build with contrib operators (verified on 1.32 CPU and CUDA).
On a Windows CUDA machine, `dotnet test src/Components/Test/VoiceBox.DNR.GRENet.Tests -c Release -p:GRENetTestCuda=true`
also checks the GPU runner against full-state reference samples across two ring wraps and reset.
