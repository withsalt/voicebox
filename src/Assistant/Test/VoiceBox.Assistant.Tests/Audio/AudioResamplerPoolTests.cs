using VoiceBox.Assistant.Realtime.Resampling;
using VoiceBox.Audio.Common.Resampling.Service;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class AudioResamplerPoolTests
{
    [Fact]
    public void Acquire_ReusesPooledResamplerAfterLeaseDisposed()
    {
        using var pool = new AudioResamplerPool(targetSampleRate: 16000, targetChannels: 1);
        AudioResampleService firstResampler;
        {
            using AudioResamplerPool.Lease first = pool.Acquire();
            firstResampler = first.Resampler;
        }

        using AudioResamplerPool.Lease second = pool.Acquire();

        Assert.Same(firstResampler, second.Resampler);
    }

    [Fact]
    public void Acquire_ReturnsSeparateResamplerWhenPooledLeaseIsActive()
    {
        using var pool = new AudioResamplerPool(targetSampleRate: 16000, targetChannels: 1);
        using AudioResamplerPool.Lease first = pool.Acquire();
        using AudioResamplerPool.Lease overlapping = pool.Acquire();

        Assert.NotSame(first.Resampler, overlapping.Resampler);
    }

    [Fact]
    public void Acquire_AfterDisposeReturnsDisposableOneShotResampler()
    {
        var pool = new AudioResamplerPool(targetSampleRate: 16000, targetChannels: 1);
        pool.Dispose();

        using AudioResamplerPool.Lease lease = pool.Acquire();

        Assert.NotNull(lease.Resampler);
    }

    [Fact]
    public void Dispose_DoesNotDisposeActivePooledLeaseUntilLeaseIsReturned()
    {
        var pool = new AudioResamplerPool(targetSampleRate: 16000, targetChannels: 1);
        AudioResamplerPool.Lease lease = pool.Acquire();

        pool.Dispose();

        lease.Resampler.Reset();
        lease.Dispose();

        Assert.Throws<ObjectDisposedException>(() => lease.Resampler.Reset());
    }
}
