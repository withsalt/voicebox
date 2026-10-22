using System.Buffers;
using System.Buffers.Binary;

namespace AudioRecordDemo;

internal sealed class Pcm16WaveWriter : IDisposable
{
    private const int HeaderSize = 44;
    private const long MaximumDataSize = uint.MaxValue - 36L;

    private readonly FileStream _stream;
    private readonly BinaryWriter _writer;
    private readonly int _sampleRate;
    private readonly short _channelCount;
    private long _dataSize;
    private bool _headerFinalized;
    private bool _disposed;

    public Pcm16WaveWriter(string path, int sampleRate, int channelCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        if (channelCount is <= 0 or > short.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(channelCount));
        }

        _sampleRate = sampleRate;
        _channelCount = (short)channelCount;
        _stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        _writer = new BinaryWriter(_stream);
        WriteHeader(dataSize: 0);
    }

    public TimeSpan Duration => TimeSpan.FromSeconds(
        _dataSize / (double)(sizeof(short) * _channelCount * _sampleRate));

    public void WriteSamples(ReadOnlySpan<float> samples)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_headerFinalized)
        {
            throw new InvalidOperationException("WAV 文件已经完成写入。");
        }

        int byteCount = checked(samples.Length * sizeof(short));
        if (_dataSize + byteCount > MaximumDataSize)
        {
            throw new IOException("WAV 文件超过 RIFF 格式支持的 4 GB 大小限制。");
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(byteCount);
        try
        {
            Span<byte> bytes = buffer.AsSpan(0, byteCount);
            for (int i = 0; i < samples.Length; i++)
            {
                float sample = float.IsFinite(samples[i]) ? Math.Clamp(samples[i], -1f, 1f) : 0f;
                short pcm = sample <= -1f
                    ? short.MinValue
                    : (short)MathF.Round(sample * short.MaxValue);
                BinaryPrimitives.WriteInt16LittleEndian(bytes.Slice(i * sizeof(short), sizeof(short)), pcm);
            }

            _writer.Write(bytes);
            _dataSize += byteCount;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    public void FinalizeHeader()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_headerFinalized)
        {
            return;
        }

        _writer.Flush();
        _stream.Position = 0;
        WriteHeader(checked((uint)_dataSize));
        _writer.Flush();
        _headerFinalized = true;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        if (!_headerFinalized)
        {
            FinalizeHeader();
        }

        _disposed = true;
        _writer.Dispose();
    }

    private void WriteHeader(uint dataSize)
    {
        const short bitsPerSample = 16;
        short blockAlign = checked((short)(_channelCount * bitsPerSample / 8));
        int byteRate = checked(_sampleRate * blockAlign);

        _writer.Write("RIFF"u8);
        _writer.Write(checked(36u + dataSize));
        _writer.Write("WAVE"u8);
        _writer.Write("fmt "u8);
        _writer.Write(16u);
        _writer.Write((short)1);
        _writer.Write(_channelCount);
        _writer.Write(_sampleRate);
        _writer.Write(byteRate);
        _writer.Write(blockAlign);
        _writer.Write(bitsPerSample);
        _writer.Write("data"u8);
        _writer.Write(dataSize);

        if (_stream.Position != HeaderSize)
        {
            throw new InvalidOperationException("生成的 WAV 文件头长度无效。");
        }
    }
}
