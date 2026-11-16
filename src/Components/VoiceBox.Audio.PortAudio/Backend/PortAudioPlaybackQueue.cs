using System.Buffers;
using System.Threading.Channels;
using VoiceBox.Audio.PortAudio.Models;

namespace VoiceBox.Audio.PortAudio.Backend;

/// <summary>Owns playback buffers independently of the duplex device's capture lifetime.</summary>
internal sealed class PortAudioPlaybackQueue : IDisposable
{
    private readonly object _readerLock = new();
    private readonly Channel<QueueBuffer> _channel;
    private QueueBuffer? _currentBuffer;
    private int _currentBufferPosition;
    private volatile bool _isPlaying;

    public PortAudioPlaybackQueue(int capacity)
    {
        _channel = Channel.CreateBounded<QueueBuffer>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false,
        });
    }

    public bool IsPlaying => _isPlaying;

    public int Count => _channel.Reader.Count;

    public bool TryEnqueue(ReadOnlySpan<float> audio, out int droppedSamples)
    {
        droppedSamples = 0;
        if (audio.IsEmpty)
            return true;

        float[] samples = ArrayPool<float>.Shared.Rent(audio.Length);
        audio.CopyTo(samples);
        var buffer = new QueueBuffer(samples, audio.Length);
        if (_channel.Writer.TryWrite(buffer))
            return true;

        buffer.Release();
        droppedSamples = audio.Length;
        return false;
    }

    /// <summary>Fills output and reports natural playback completion once.</summary>
    public bool Fill(Span<float> output)
    {
        // Clearing playback must never stall the duplex callback's microphone processing.
        if (!Monitor.TryEnter(_readerLock))
        {
            output.Clear();
            return false;
        }

        try
        {
            int written = 0;
            while (written < output.Length)
            {
                if (_currentBuffer is not null && _currentBufferPosition >= _currentBuffer.Length)
                {
                    _currentBuffer.Release();
                    _currentBuffer = null;
                    _currentBufferPosition = 0;
                }

                if (_currentBuffer is null)
                {
                    if (!_channel.Reader.TryRead(out _currentBuffer))
                    {
                        output[written..].Clear();
                        bool completed = _isPlaying;
                        _isPlaying = false;
                        return completed;
                    }
                    _currentBufferPosition = 0;
                    _isPlaying = true;
                }

                int count = Math.Min(_currentBuffer.Length - _currentBufferPosition, output.Length - written);
                _currentBuffer.Data.AsSpan(_currentBufferPosition, count).CopyTo(output[written..]);
                _currentBufferPosition += count;
                written += count;
            }
            return false;
        }
        finally
        {
            Monitor.Exit(_readerLock);
        }
    }

    public void Clear()
    {
        lock (_readerLock)
        {
            _currentBuffer?.Release();
            _currentBuffer = null;
            _currentBufferPosition = 0;
            while (_channel.Reader.TryRead(out QueueBuffer? buffer))
                buffer.Release();
            _isPlaying = false;
        }
    }

    public void Complete() => _channel.Writer.TryComplete();

    public void Dispose()
    {
        Complete();
        Clear();
    }
}
