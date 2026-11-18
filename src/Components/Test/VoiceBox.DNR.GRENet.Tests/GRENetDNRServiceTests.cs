using System.Diagnostics;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.DNR.GRENet;
using VoiceBox.DNR.GRENet.Models;
using Xunit;

namespace VoiceBox.DNR.GRENet.Tests;

public sealed class GRENetDNRServiceTests
{
    [Fact]
    public void ProcessAllReturnsFiniteSourceLengthForArbitraryChunks()
    {
        float[] input = CreateSignal(16000 / 2);
        using var service = CreateService();

        float[] output = service.ProcessAll(input, chunkSize: 137);

        Assert.Equal(input.Length, output.Length);
        Assert.All(output, sample => Assert.True(float.IsFinite(sample)));
        Assert.Contains(output, sample => MathF.Abs(sample) > 1e-5f);
    }

    [Fact]
    public void ProcessingIsStableAcrossChunkBoundaries()
    {
        float[] input = CreateSignal(16000 / 4);
        using var first = CreateService();
        using var second = CreateService();

        float[] firstOutput = first.ProcessAll(input, chunkSize: 200);
        float[] secondOutput = second.ProcessAll(input, chunkSize: 73);

        Assert.Equal(firstOutput.Length, secondOutput.Length);
        for (int i = 400; i < firstOutput.Length; i++)
        {
            Assert.InRange(MathF.Abs(firstOutput[i] - secondOutput[i]), 0f, 2e-3f);
        }
    }

    [Fact]
    public void ResetStartsAnIndependentStream()
    {
        float[] input = CreateSignal(16000 / 5);
        using var service = CreateService();

        float[] first = service.ProcessAll(input, chunkSize: 200);
        service.Reset();
        float[] second = service.ProcessAll(input, chunkSize: 200);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.InRange(MathF.Abs(first[i] - second[i]), 0f, 1e-5f);
        }
    }

    [Theory]
    [InlineData(OnnxExecutionProvider.Cpu)]
#if GRENET_CUDA_TESTS
    [InlineData(OnnxExecutionProvider.Cuda)]
#endif
    public void BundledModelMatchesFullStateReferenceAcrossRingWrapsAndReset(OnnxExecutionProvider provider)
    {
        var options = new GRENetOptions
        {
            ExecutionProvider = provider,
        };
        using var service = new GRENetDNRService(Options.Create(options));
        float[] input = CreateSignal(210 * service.FrameSize);
        var noise = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] += 0.04f * (noise.NextSingle() - 0.5f);
        }

        // Eight complete reference hops from the previous full-state CPU model, two threads, with the same input.
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "wrap-reference.f32");
        using var reader = new BinaryReader(File.OpenRead(path));
        float[] first = service.ProcessAll(input, chunkSize: 200);
        float[] afterReset = service.ProcessAll(input, chunkSize: 137);
        const float waveformTolerance = 1e-5f;
        int[] hops = [97, 98, 99, 100, 197, 198, 199, 200];
        foreach (int hop in hops)
        {
            for (int i = hop * service.FrameSize; i < (hop + 1) * service.FrameSize; i++)
            {
                float expected = reader.ReadSingle();
                Assert.InRange(MathF.Abs(expected - first[i]), 0f, waveformTolerance);
                Assert.InRange(MathF.Abs(expected - afterReset[i]), 0f, waveformTolerance);
            }
        }
    }

    [Fact]
    public void RejectsUnsupportedAudioFormat()
    {
        GRENetOptions options = new() { SampleRate = 8000 };
        Assert.Throws<NotSupportedException>(() => new GRENetDNRService(Options.Create(options)));
    }

    [Fact]
    public void RejectsUnsupportedExecutionProvider()
    {
        GRENetOptions options = new() { ExecutionProvider = VoiceBox.Core.Onnx.OnnxExecutionProvider.DirectML };
        Assert.Throws<NotSupportedException>(() => new GRENetDNRService(Options.Create(options)));
    }

    [Fact]
    public void CpuThreadsDefaultToTwo()
    {
        var options = new GRENetOptions();
        Assert.Empty(options.Model);
        Assert.Equal(2, options.ThreadsNum);

        options.Model = "custom.onnx";
        Assert.Equal(2, options.ThreadsNum);

        options.ThreadsNum = 3;
        options.Model = string.Empty;
        Assert.Equal(3, options.ThreadsNum);
    }

    [Fact]
    public void MissingCustomModelReportsTheConfiguredPath()
    {
        string path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.onnx");
        var options = new GRENetOptions
        {
            Model = path,
        };

        var error = Assert.Throws<FileNotFoundException>(() => new GRENetDNRService(Options.Create(options)));
        Assert.Equal(path, error.FileName);
    }

    [Fact]
    public void OneSecondStreamCompletesWithinRealtimeSafetyBudget()
    {
        float[] input = CreateSignal(16000);
        using var service = CreateService();
        Stopwatch stopwatch = Stopwatch.StartNew();

        _ = service.ProcessAll(input, chunkSize: service.FrameSize);

        stopwatch.Stop();
        Assert.True(stopwatch.Elapsed < TimeSpan.FromSeconds(30), $"GRE-Net processing took {stopwatch.Elapsed}.");
    }

    private static GRENetDNRService CreateService() => new(Options.Create(new GRENetOptions()));

    private static float[] CreateSignal(int length)
    {
        var signal = new float[length];
        for (int i = 0; i < signal.Length; i++)
        {
            float t = i / 16000f;
            signal[i] = 0.15f * MathF.Sin(2f * MathF.PI * 220f * t) + 0.03f * MathF.Sin(2f * MathF.PI * 1270f * t);
        }

        return signal;
    }
}
