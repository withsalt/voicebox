#:property TargetFramework=net10.0
#:property PublishAot=false
#:package Microsoft.ML.OnnxRuntime@1.32.0

using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;

if (args.Length < 3)
{
    Console.Error.WriteLine("Usage: bench_compare.cs <threads> <frames> <model>...");
    return 1;
}

int threads = int.Parse(args[0]);
int framesPerRound = int.Parse(args[1]);
var paths = args.Skip(2).ToArray();
const int rounds = 6;

var runners = paths.Select(path => new Runner(path, threads)).ToArray();
var samples = new float[(rounds * framesPerRound + 2) * 200];
var random = new Random(42);
for (int i = 0; i < samples.Length; i++)
{
    double t = (double)i / 16000;
    samples[i] = (float)(0.12 * Math.Sin(2 * Math.PI * (140 * t + 45 * t * t)) + 0.025 * (2 * random.NextDouble() - 1));
}

var roundsByModel = runners.Select(_ => new List<double>()).ToArray();
for (int round = 0; round < rounds; round++)
{
    var order = round % 2 == 0 ? Enumerable.Range(0, runners.Length) : Enumerable.Range(0, runners.Length).Reverse();
    foreach (int index in order)
    {
        Runner runner = runners[index];
        var times = new double[framesPerRound];
        for (int frame = 0; frame < framesPerRound; frame++)
        {
            int offset = (round * framesPerRound + frame) * 200;
            long start = Stopwatch.GetTimestamp();
            runner.Step(samples.AsSpan(offset, 200));
            times[frame] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }

        Array.Sort(times);
        roundsByModel[index].Add(times[times.Length / 2]);
    }
}

for (int index = 0; index < runners.Length; index++)
{
    var values = roundsByModel[index].OrderBy(value => value).ToArray();
    Console.WriteLine($"{Path.GetFileName(paths[index]),-32} median {values[values.Length / 2]:F3} ms  rounds [{string.Join(", ", values.Select(v => v.ToString("F3")))}]");
}

return 0;

sealed class Runner
{
    private readonly InferenceSession _session;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly OrtValue[] _state;
    private readonly OrtValue[] _inputValues;
    private readonly float[][] _buffers;
    private readonly float[] _amplitude = new float[201];
    private readonly float[] _phase = new float[201];
    private readonly RunOptions _runOptions = new();
    private readonly int _history;
    private IDisposableReadOnlyCollection<OrtValue>? _previous;
    private int _frame;

    public Runner(string path, int threads)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = threads,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            EnableMemoryPattern = true,
            EnableCpuMemArena = true,
        };
        _session = new InferenceSession(path, options);
        var inputs = _session.InputMetadata;
        _inputNames = inputs.Keys.ToArray();
        _outputNames = _session.OutputMetadata.Keys.ToArray();
        int cacheCount = _inputNames.Length - 2;
        _state = new OrtValue[cacheCount];
        _buffers = new float[cacheCount][];
        for (int i = 0; i < cacheCount; i++)
        {
            long[] shape = inputs[_inputNames[i + 2]].Dimensions.Select(d => (long)d).ToArray();
            var buffer = new float[shape.Aggregate(1L, (a, b) => a * b)];
            _buffers[i] = buffer;
            _state[i] = OrtValue.CreateTensorValueFromMemory(buffer, shape);
        }

        _history = (int)inputs[_inputNames[Array.FindIndex(_inputNames, name => name.StartsWith("ts_k_cache", StringComparison.Ordinal))]].Dimensions[^2];
        _inputValues = new OrtValue[_inputNames.Length];
        _inputValues[0] = OrtValue.CreateTensorValueFromMemory(_amplitude, new long[] { 1, 201, 1 });
        _inputValues[1] = OrtValue.CreateTensorValueFromMemory(_phase, new long[] { 1, 201, 1 });
        Array.Copy(_state, 0, _inputValues, 2, cacheCount);
        for (int i = 0; i < 80; i++)
        {
            Step(new float[200]);
        }
    }

    public void Step(ReadOnlySpan<float> hop)
    {
        // The benchmark signal is irrelevant to shape; fill deterministic spectra.
        for (int i = 0; i < 201; i++)
        {
            _amplitude[i] = 0.2f + 0.1f * ((i * 7 + _frame) % 13) / 13f;
            _phase[i] = -1.5f + 3f * ((i * 5 + _frame) % 17) / 17f;
        }

        var results = _session.Run(_runOptions, _inputNames, _inputValues, _outputNames);
        int slot = _frame % _history;
        for (int i = 0; i < _state.Length; i++)
        {
            OrtValue next;
            if (_outputNames[i + 3].EndsWith("_frame", StringComparison.Ordinal))
            {
                ReadOnlySpan<float> data = results[i + 3].GetTensorDataAsSpan<float>();
                float[] ring = _buffers[i];
                int width = ring.Length / _history;
                int rows = ring.Length / (_history * width);
                for (int row = 0; row < rows; row++)
                {
                    data.Slice(row * width, width).CopyTo(ring.AsSpan((row * _history + slot) * width, width));
                }

                next = _state[i];
            }
            else
            {
                next = results[i + 3];
            }

            _inputValues[i + 2] = next;
            _state[i] = next;
        }

        _previous?.Dispose();
        _previous = results;
        _frame++;
    }
}
