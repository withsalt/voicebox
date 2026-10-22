using VoiceBox.Audio.Common.Resampling.Service;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class AudioResampleServiceTests
{
    // ── 抗混叠：高于目标 Nyquist 的分量应被强烈抑制，而通带分量基本保留 ──────────────

    [Fact]
    public void Downsample_AttenuatesAboveTargetNyquist()
    {
        const int srcRate = 24000;
        const int dstRate = 16000; // 目标 Nyquist = 8000Hz
        const int n = srcRate;     // 1 秒

        // 11kHz 远高于 8kHz 目标 Nyquist：理想带限重采样应几乎完全去除 (线性插值则会混叠到 5kHz)。
        float[] input = Tone(11000, srcRate, n);

        using var svc = new AudioResampleService(new AudioResampleOptions
        {
            TargetSampleRate = dstRate,
            TargetChannels = 1,
        });

        float[] output = ConvertAll(svc, input, srcRate, 1);

        double inRms = Rms(input, 0, input.Length);
        double outRms = RmsMiddle(output);

        // 至少 40dB 抑制 (ratio < 0.01)。
        Assert.True(outRms < inRms * 0.01, $"aliasing not suppressed: inRms={inRms:F4} outRms={outRms:F4}");
    }

    [Fact]
    public void Downsample_PreservesPassbandAmplitude()
    {
        const int srcRate = 24000;
        const int dstRate = 16000;
        const int n = srcRate;

        float[] input = Tone(1000, srcRate, n); // 通带内

        using var svc = new AudioResampleService(new AudioResampleOptions
        {
            TargetSampleRate = dstRate,
            TargetChannels = 1,
        });

        float[] output = ConvertAll(svc, input, srcRate, 1);

        double inRms = Rms(input, 0, input.Length);
        double outRms = RmsMiddle(output);
        double ratio = outRms / inRms;

        // 显式增益归一化后通带应非常平直 (±2%)。
        Assert.InRange(ratio, 0.98, 1.02);

        // 采样率比例应反映在输出长度上 (16000/24000)。
        double expected = n * (double)dstRate / srcRate;
        Assert.InRange(output.Length, expected - 50, expected + 50);
    }

    [Fact]
    public void Downsample_UnityDcGain()
    {
        const int srcRate = 24000;
        const int dstRate = 16000;
        const int n = srcRate;

        // 直流 (常量) 输入：单位增益重采样后稳态输出应等于同一常量。
        const float dc = 0.5f;
        var input = new float[n];
        Array.Fill(input, dc);

        using var svc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = dstRate, TargetChannels = 1 });
        float[] output = ConvertAll(svc, input, srcRate, 1);

        // 取中段稳态比对，跳过群延迟/冲洗瞬态。
        int start = output.Length / 4;
        int end = output.Length * 3 / 4;
        for (int i = start; i < end; i++)
            Assert.Equal(dc, output[i], 3); // 误差 < 1e-3
    }

    // ── 块边界相位连续性：整段一次 vs 拆成小块，结果应一致 ──────────────────────────

    [Fact]
    public void ChunkedConversion_MatchesSingleShot()
    {
        const int srcRate = 24000;
        const int dstRate = 16000;
        const int n = 4800;

        float[] input = MixedTones(srcRate, n);

        using var whole = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = dstRate, TargetChannels = 1 });
        float[] outWhole = ConvertAll(whole, input, srcRate, 1);

        using var chunked = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = dstRate, TargetChannels = 1 });
        var acc = new List<float>();
        const int chunk = 137; // 故意取非整除块长
        for (int i = 0; i < input.Length; i += chunk)
        {
            int len = Math.Min(chunk, input.Length - i);
            acc.AddRange(chunked.Convert(input.AsSpan(i, len), srcRate, 1));
        }
        acc.AddRange(chunked.Flush());
        float[] outChunked = acc.ToArray();

        Assert.True(Math.Abs(outWhole.Length - outChunked.Length) <= 2,
            $"length mismatch: whole={outWhole.Length} chunked={outChunked.Length}");

        int compare = Math.Min(outWhole.Length, outChunked.Length);
        double maxDiff = 0;
        for (int i = 0; i < compare; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(outWhole[i] - outChunked[i]));

        Assert.True(maxDiff < 1e-4, $"chunk boundary discontinuity: maxDiff={maxDiff:E3}");
    }

    // ── 声道转换 ────────────────────────────────────────────────────────────────

    [Fact]
    public void StereoToMono_AveragesChannels()
    {
        const int rate = 16000;
        // 帧序列: L=0.4, R=0.8 → mono 应为 0.6。
        int frames = 100;
        var input = new float[frames * 2];
        for (int f = 0; f < frames; f++)
        {
            input[f * 2] = 0.4f;
            input[f * 2 + 1] = 0.8f;
        }

        using var svc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = rate, TargetChannels = 1 });
        float[] output = svc.Convert(input, rate, 2); // 等采样率，仅声道处理

        Assert.Equal(frames, output.Length);
        foreach (float v in output)
            Assert.Equal(0.6f, v, 4);
    }

    [Fact]
    public void MonoToStereo_DuplicatesChannel()
    {
        const int rate = 16000;
        var input = new float[] { 0.1f, 0.2f, 0.3f };

        using var svc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = rate, TargetChannels = 2 });
        float[] output = svc.Convert(input, rate, 1);

        Assert.Equal(6, output.Length);
        for (int f = 0; f < input.Length; f++)
        {
            Assert.Equal(input[f], output[f * 2], 5);
            Assert.Equal(input[f], output[f * 2 + 1], 5);
        }
    }

    [Fact]
    public void MonoToStereo_WithResampling_DuplicatesResampledChannels()
    {
        const int srcRate = 24000;
        const int dstRate = 16000;
        const int n = 2400;

        float[] input = MixedTones(srcRate, n);

        using var svc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = dstRate, TargetChannels = 2 });
        float[] output = ConvertAll(svc, input, srcRate, 1);

        Assert.Equal(0, output.Length % 2);
        double expectedFrames = n * (double)dstRate / srcRate;
        Assert.InRange(output.Length / 2, expectedFrames - 50, expectedFrames + 50);

        for (int frame = 0; frame < output.Length / 2; frame++)
            Assert.Equal(output[frame * 2], output[frame * 2 + 1], 6);
    }

    // ── 等采样率直通 ────────────────────────────────────────────────────────────

    [Fact]
    public void EqualFormat_PassesThrough()
    {
        const int rate = 16000;
        var input = new float[] { -0.5f, 0.25f, 0.75f, -1.0f, 0.0f };

        using var svc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = rate, TargetChannels = 1 });
        float[] output = svc.Convert(input, rate, 1);

        Assert.Equal(input, output);
    }

    // ── 来源格式兜底 ────────────────────────────────────────────────────────────

    [Fact]
    public void Convert_UsesConfiguredSourceFormat_WhenNotSpecified()
    {
        using var svc = new AudioResampleService(new AudioResampleOptions
        {
            TargetSampleRate = 16000,
            TargetChannels = 1,
            SourceSampleRate = 24000,
            SourceChannels = 1,
        });

        float[] input = Tone(1000, 24000, 24000);
        float[] output = ConvertAll(svc, input, 0, 0); // 不传来源格式 → 用配置兜底

        double expected = 24000 * (16000.0 / 24000.0);
        Assert.InRange(output.Length, expected - 50, expected + 50);
    }

    [Fact]
    public void Convert_Throws_WhenSourceFormatUnknown()
    {
        using var svc = new AudioResampleService(new AudioResampleOptions
        {
            TargetSampleRate = 16000,
            TargetChannels = 1,
            // 未配置 SourceSampleRate/Channels
        });

        var input = new float[100];
        Assert.Throws<InvalidOperationException>(() => svc.Convert(input, 0, 0));
    }

    // ── ref-buffer (零分配) API ───────────────────────────────────────────────────

    [Fact]
    public void RefApi_MatchesAllocating_AcrossReusedBuffer()
    {
        const int srcRate = 24000;
        const int dstRate = 16000;
        const int chunk = 137; // 非整除块长
        float[] input = MixedTones(srcRate, 5000);

        // 基准：分配版逐块 (与整段一致性已由其它测试保证)。
        using var alloc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = dstRate, TargetChannels = 1 });
        var expected = new List<float>();
        for (int i = 0; i < input.Length; i += chunk)
            expected.AddRange(alloc.Convert(input.AsSpan(i, Math.Min(chunk, input.Length - i)), srcRate, 1));
        expected.AddRange(alloc.Flush());

        // ref 版：单一复用缓冲，从极小起始以验证扩容。
        using var refSvc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = dstRate, TargetChannels = 1 });
        var actual = new List<float>();
        float[] buf = new float[1];
        for (int i = 0; i < input.Length; i += chunk)
        {
            int n = refSvc.Convert(input.AsSpan(i, Math.Min(chunk, input.Length - i)), srcRate, 1, ref buf);
            actual.AddRange(buf.AsSpan(0, n).ToArray());
        }
        int m = refSvc.Flush(ref buf);
        actual.AddRange(buf.AsSpan(0, m).ToArray());

        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
            Assert.Equal(expected[i], actual[i], 6);
    }

    [Fact]
    public void RefApi_ReusedBuffer_LargeThenSmall_OnlyCountIsValid()
    {
        using var svc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = 16000, TargetChannels = 1 });

        float[] buf = Array.Empty<float>();
        float[] big = Tone(1000, 16000, 16000); // 等采样率直通，count=16000，缓冲被撑大
        int n1 = svc.Convert(big, 16000, 1, ref buf);
        Assert.Equal(16000, n1);

        // 紧接一个小块：返回值只应为 4，且 [0,4) 内容正确 (不读 count 之外的陈旧数据)。
        float[] small = [0.1f, 0.2f, 0.3f, 0.4f];
        int n2 = svc.Convert(small, 16000, 1, ref buf);
        Assert.Equal(4, n2);
        for (int i = 0; i < 4; i++)
            Assert.Equal(small[i], buf[i], 6);
    }

    [Fact]
    public void Flush_AfterTailIsDrained_IsIdempotent()
    {
        using var svc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = 16000, TargetChannels = 1 });

        float[] buf = Array.Empty<float>();
        int converted = svc.Convert(Tone(1000, 48000, 480), 48000, 1, ref buf);
        int firstFlush = svc.Flush(ref buf);
        int secondFlush = svc.Flush(ref buf);

        Assert.True(converted > 0);
        Assert.True(firstFlush > 0);
        Assert.Equal(0, secondFlush);
    }

    [Fact]
    public void RefApi_SteadyState_NoPerChunkAllocation()
    {
        using var svc = new AudioResampleService(new AudioResampleOptions { TargetSampleRate = 16000, TargetChannels = 1 });
        float[] buf = new float[8192];
        float[] chunk = Tone(1000, 48000, 1024); // 48k -> 16k

        // 预热：建表、缓冲扩容到稳态、JIT。
        for (int i = 0; i < 100; i++) svc.Convert(chunk, 48000, 1, ref buf);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
            svc.Convert(chunk, 48000, 1, ref buf);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // 稳态下逐块 Convert 应零分配 (留极小余量以容忍偶发计数噪声)。
        Assert.True(allocated < 4096, $"steady-state allocated {allocated} bytes over 1000 convert calls");
    }

    // ── helpers ─────────────────────────────────────────────────────────────────

    private static float[] Tone(double freq, int rate, int n)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++)
            x[i] = (float)Math.Sin(2.0 * Math.PI * freq * i / rate);
        return x;
    }

    private static float[] MixedTones(int rate, int n)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++)
            x[i] = (float)(0.5 * Math.Sin(2.0 * Math.PI * 500 * i / rate)
                         + 0.3 * Math.Sin(2.0 * Math.PI * 2000 * i / rate));
        return x;
    }

    private static float[] ConvertAll(IAudioResampleService svc, float[] input, int srcRate, int srcCh)
    {
        var acc = new List<float>();
        acc.AddRange(svc.Convert(input, srcRate, srcCh));
        acc.AddRange(svc.Flush());
        return acc.ToArray();
    }

    private static double Rms(float[] x, int start, int count)
    {
        double sum = 0;
        for (int i = start; i < start + count; i++)
            sum += (double)x[i] * x[i];
        return Math.Sqrt(sum / Math.Max(1, count));
    }

    /// <summary>取中间 50% 测 RMS，跳过起止的群延迟/冲洗瞬态。</summary>
    private static double RmsMiddle(float[] x)
    {
        if (x.Length < 8) return Rms(x, 0, x.Length);
        int start = x.Length / 4;
        int count = x.Length / 2;
        return Rms(x, start, count);
    }
}
