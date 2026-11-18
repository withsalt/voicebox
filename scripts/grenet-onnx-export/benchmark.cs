#:property TargetFramework=net10.0
#:property PublishAot=false
#:project ../../src/Components/VoiceBox.DNR.GRENet/VoiceBox.DNR.GRENet.csproj
#:package Microsoft.ML.OnnxRuntime.Gpu@1.32.0

using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.DNR.GRENet;
using VoiceBox.DNR.GRENet.Models;

if (args.Length is < 1 or > 4)
{
    Console.Error.WriteLine("Usage: benchmark.cs <model.onnx> [Cpu|Cuda] [threads] [chunkSamples]");
    return 1;
}

var provider = args.Length > 1 ? Enum.Parse<OnnxExecutionProvider>(args[1], ignoreCase: true) : OnnxExecutionProvider.Cpu;
int threads = args.Length > 2 ? int.Parse(args[2]) : provider == OnnxExecutionProvider.Cpu ? 2 : 1;
int chunk = args.Length > 3 ? int.Parse(args[3]) : 200;
int seconds = int.TryParse(Environment.GetEnvironmentVariable("GRENET_SECONDS"), out int configuredSeconds) ? configuredSeconds : 10;
ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threads);
ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunk);
ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seconds);
var options = new GRENetOptions
{
    Model = Path.GetFullPath(args[0]),
    ExecutionProvider = provider,
    ThreadsNum = threads,
};
bool useTf32 = Environment.GetEnvironmentVariable("GRENET_CUDA_TF32") == "1";
if (provider == OnnxExecutionProvider.Cuda)
{
    options.CudaTuning = new CudaExecutionTuning
    {
        AdditionalOptions = new()
        {
            ["enable_cuda_graph"] = "1",
            ["use_tf32"] = useTf32 ? "1" : "0",
        },
    };
}
string? profilePath = Environment.GetEnvironmentVariable("GRENET_PROFILE");
if (!string.IsNullOrWhiteSpace(profilePath))
{
    options.SessionOptionsFactory = () =>
    {
        SessionOptions session = provider == OnnxExecutionProvider.Cpu
            ? OnnxSessionFactory.CreateCpu(threads)
            : OnnxSessionFactory.CreateCuda(options.GpuDeviceId, threads, options.CudaTuning, SessionPurpose.StaticShapes);
        session.EnableProfiling = true;
        session.AddSessionConfigEntry("session.profile_file_prefix", Path.GetFullPath(profilePath));
        return session;
    };
}

const int sampleRate = 16000;
using var service = new GRENetDNRService(Options.Create(options));
int warmupCalls = (int)Math.Ceiling(120d * service.FrameSize / chunk);
int calls = (int)Math.Ceiling((double)seconds * sampleRate / chunk);
var samples = new float[checked((warmupCalls + calls) * chunk)];
var random = new Random(42);
for (int i = 0; i < samples.Length; i++)
{
    double t = (double)i / sampleRate;
    samples[i] = (float)(0.12 * Math.Sin(2 * Math.PI * (140 * t + 45 * t * t)) + 0.025 * (2 * random.NextDouble() - 1));
}

var times = new double[calls];
for (int i = 0; i < warmupCalls + calls; i++)
{
    Span<float> buffer = samples.AsSpan(i * chunk, chunk);
    long start = Stopwatch.GetTimestamp();
    service.Process(buffer);
    if (i >= warmupCalls)
    {
        times[i - warmupCalls] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
    }
}

Array.Sort(times);
double budgetMs = 1000d * chunk / sampleRate;
double meanMs = times.Average();
Console.WriteLine(JsonSerializer.Serialize(new
{
    Runtime = OrtEnv.Instance().GetVersionString(),
    Provider = provider.ToString(),
    CudaTf32 = provider == OnnxExecutionProvider.Cuda ? useTf32 : (bool?)null,
    Threads = threads,
    Chunk = chunk,
    Calls = calls,
    MeanMs = meanMs,
    P99Ms = times[(int)Math.Ceiling(calls * 0.99) - 1],
    Rtf = meanMs / budgetMs,
    OverBudget = times.Count(ms => ms > budgetMs),
}));
return 0;
