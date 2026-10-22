using System.Buffers;
using System.Threading.Channels;

namespace VoiceBox.Audio.MiniAudio.Backend;

internal sealed class MiniAudioPlaybackQueue : IDisposable
{
    private readonly Channel<MiniAudioQueueBuffer> _channel;
    private MiniAudioQueueBuffer _currentBuffer;
    private int _currentBufferPosition;
    private volatile bool _isPlaying;

    internal MiniAudioPlaybackQueue(uint capacity)
    {
        _channel = Channel.CreateBounded<MiniAudioQueueBuffer>(new BoundedChannelOptions((int)capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false,
        });
    }

    internal bool IsPlaying => _isPlaying;

    internal int Count => _channel.Reader.Count;

    internal bool TryEnqueue(ReadOnlySpan<float> data, out int droppedSamples)
    {
        droppedSamples = 0;
        if (data.IsEmpty)
        {
            return true;
        }

        // Round up to power of 2 for better ArrayPool bucketing
        int requestedSize = data.Length;
        int rentSize = requestedSize <= 16 ? requestedSize : RoundUpToPowerOf2(requestedSize);
        float[] buffer = ArrayPool<float>.Shared.Rent(rentSize);
        data.CopyTo(buffer);

        if (_channel.Writer.TryWrite(new MiniAudioQueueBuffer(buffer, requestedSize)))
        {
            return true;
        }

        ArrayPool<float>.Shared.Return(buffer, clearArray: true);
        droppedSamples = requestedSize;
        return false;
    }

    private static int RoundUpToPowerOf2(int value)
    {
        // For values > 16, round up to next power of 2 for better pooling
        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        return value + 1;
    }

    internal bool Fill(Span<float> output)
    {
        int outputLength = output.Length;
        int samplesWritten = 0;

        while (samplesWritten < outputLength)
        {
            if (_currentBuffer.HasData && _currentBufferPosition >= _currentBuffer.Length)
            {
                _currentBuffer.Release();
                _currentBufferPosition = 0;
            }

            if (!_currentBuffer.HasData)
            {
                if (!_channel.Reader.TryRead(out _currentBuffer))
                {
                    // Clear remaining output
                    if (samplesWritten < outputLength)
                    {
                        output.Slice(samplesWritten).Clear();
                    }

                    bool completed = _isPlaying;
                    _isPlaying = false;
                    return completed;
                }

                _currentBufferPosition = 0;
                _isPlaying = true;
            }

            int available = _currentBuffer.Length - _currentBufferPosition;
            int remaining = outputLength - samplesWritten;
            int copyCount = Math.Min(available, remaining);

            // Direct span copy without intermediate slicing
            _currentBuffer.Data.AsSpan(_currentBufferPosition, copyCount)
                .CopyTo(output.Slice(samplesWritten, copyCount));

            _currentBufferPosition += copyCount;
            samplesWritten += copyCount;
        }

        return false;
    }

    internal void Clear()
    {
        _currentBuffer.Release();
        _currentBufferPosition = 0;
        while (_channel.Reader.TryRead(out MiniAudioQueueBuffer buffer))
        {
            buffer.Release();
        }

        _isPlaying = false;
    }

    public void Dispose()
    {
        _channel.Writer.TryComplete();
        Clear();
    }
}

// A descriptor has one owner: the writer until TryWrite succeeds, then the reader.
// Clear/Dispose only run after the device callback has stopped.
internal struct MiniAudioQueueBuffer(float[] data, int length)
{
    private float[]? _data = data;

    internal readonly float[] Data => _data!;

    internal readonly bool HasData => _data is not null;

    internal int Length { get; } = length;

    internal void Release()
    {
        float[]? data = _data;
        _data = null;
        if (data is not null)
        {
            ArrayPool<float>.Shared.Return(data, clearArray: true);
        }
    }
}
