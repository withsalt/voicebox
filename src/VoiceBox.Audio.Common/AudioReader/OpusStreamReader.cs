using Concentus;
using Concentus.Oggfile;
using System.Text;

namespace VoiceBox.Audio.Common.AudioReader;

/// <summary>
/// 流式 Opus 音频读取器。支持标准 Ogg Opus，以及每个 Opus 包前带 16 位小端长度的旧格式。
/// </summary>
public class OpusStreamReader : IAudioStreamReader
{
    private static readonly byte[] OggSignature = "OggS"u8.ToArray();

    private readonly Stream _stream;
    private readonly BinaryReader? _reader;
    private readonly IOpusDecoder _decoder;
    private readonly int _sampleRate;
    private readonly int _channels;
    private readonly int _frameSizeMs;
    private readonly long _streamPosition;
    private readonly bool _leaveOpen;
    private readonly bool _isOgg;
    private readonly bool _usesManagedDecoder;

    private OpusOggReadStream? _oggReader;
    private byte[] _packetBuffer = [];
    private short[] _shortDecodeBuffer = [];
    private float[] _decodeBuffer = [];
    private int _decodedSampleCount;
    private int _remainingSamplesOffset;
    private bool _disposed;

    /// <summary>获取解码输出采样率。</summary>
    public int SampleRate => _sampleRate;

    /// <summary>获取解码输出声道数。</summary>
    public int Channels => _channels;

    /// <summary>获取旧的长度前缀格式所使用的帧时长。</summary>
    public int FrameSizeMs => _frameSizeMs;

    /// <summary>获取旧的长度前缀格式中每帧的交错样本数。</summary>
    public int SamplesPerFrame => _sampleRate * _frameSizeMs / 1000 * _channels;

    public OpusStreamReader(string filePath, int sampleRate = 48000, int channels = 1, int frameSizeMs = 20)
        : this(OpenValidatedFile(filePath, sampleRate, channels, frameSizeMs), false, sampleRate, channels, frameSizeMs)
    {
    }

    public OpusStreamReader(
        Stream stream,
        bool leaveOpen = true,
        int sampleRate = 48000,
        int channels = 1,
        int frameSizeMs = 20)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ValidateParameters(sampleRate, channels, frameSizeMs);

        if (!stream.CanRead)
        {
            throw new ArgumentException("The Opus stream must be readable.", nameof(stream));
        }

        // 旧实现同样依赖 Position/Length；显式校验可为调用方提供更清楚的错误。
        if (!stream.CanSeek)
        {
            throw new ArgumentException("The Opus stream must support seeking.", nameof(stream));
        }

        _stream = stream;
        _sampleRate = sampleRate;
        _channels = channels;
        _frameSizeMs = frameSizeMs;
        _streamPosition = stream.Position;
        _leaveOpen = leaveOpen;

        try
        {
            _decoder = OpusCodecFactory.CreateDecoder(sampleRate, channels, TextWriter.Null);
            _usesManagedDecoder = _decoder is Concentus.Structs.OpusDecoder;
            _isOgg = HasOggSignature(stream);

            if (_isOgg)
            {
                InitializeOggReader();
            }
            else
            {
                // 统一由 OpusStreamReader 管理底层流的所有权。
                _reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            }
        }
        catch
        {
            (_decoder as IDisposable)?.Dispose();
            if (!leaveOpen)
            {
                stream.Dispose();
            }

            throw;
        }
    }

    /// <summary>从当前位置读取最多 <paramref name="buffer"/>.Length 个交错浮点样本。</summary>
    public int ReadSamples(Span<float> buffer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (buffer.IsEmpty)
        {
            return 0;
        }

        int samplesCopied = 0;

        while (samplesCopied < buffer.Length)
        {
            if (_remainingSamplesOffset < _decodedSampleCount)
            {
                int count = Math.Min(
                    _decodedSampleCount - _remainingSamplesOffset,
                    buffer.Length - samplesCopied);

                _decodeBuffer.AsSpan(_remainingSamplesOffset, count)
                    .CopyTo(buffer[samplesCopied..]);

                samplesCopied += count;
                _remainingSamplesOffset += count;

                if (_remainingSamplesOffset == _decodedSampleCount)
                {
                    _decodedSampleCount = 0;
                    _remainingSamplesOffset = 0;
                }

                continue;
            }

            int decodedSampleCount = _isOgg
                ? DecodeNextOggPacket()
                : DecodeNextLengthPrefixedPacket();

            if (decodedSampleCount < 0)
            {
                break;
            }

            _decodedSampleCount = decodedSampleCount;
            _remainingSamplesOffset = 0;
        }

        return samplesCopied;
    }

    /// <summary>将读取位置和 Opus 解码器状态重置到输入音频的起点。</summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        _decodedSampleCount = 0;
        _remainingSamplesOffset = 0;

        if (_isOgg)
        {
            _oggReader?.Close();
            _stream.Position = _streamPosition;
            _decoder.ResetState();
            InitializeOggReader();
        }
        else
        {
            _stream.Position = _streamPosition;
            _decoder.ResetState();
        }
    }

    private int DecodeNextLengthPrefixedPacket()
    {
        if (_stream.Position + sizeof(ushort) > _stream.Length)
        {
            return -1;
        }

        ushort packetLength = _reader!.ReadUInt16();
        if (_stream.Position + packetLength > _stream.Length)
        {
            throw new InvalidDataException("The length-prefixed Opus packet is truncated.");
        }

        EnsureCapacity(ref _packetBuffer, packetLength);
        ReadPacket(packetLength);
        int frameSize = _sampleRate * _frameSizeMs / 1000;
        int outputCapacity = frameSize * _channels;
        EnsureCapacity(ref _decodeBuffer, outputCapacity);
        int decodedFrames;
        if (_usesManagedDecoder)
        {
            // Concentus' managed float overload allocates this same temporary short[]
            // on every call, then applies this exact conversion factor.
            EnsureCapacity(ref _shortDecodeBuffer, outputCapacity);
            decodedFrames = _decoder.Decode(
                _packetBuffer.AsSpan(0, packetLength),
                _shortDecodeBuffer.AsSpan(0, outputCapacity),
                frameSize,
                false);
            int sampleCount = decodedFrames * _channels;
            const float shortToFloat = 3.0517578E-05f;
            for (int i = 0; i < sampleCount; i++)
            {
                _decodeBuffer[i] = shortToFloat * _shortDecodeBuffer[i];
            }
        }
        else
        {
            // Native float decoding is retained to avoid introducing integer quantization.
            decodedFrames = _decoder.Decode(
                _packetBuffer.AsSpan(0, packetLength),
                _decodeBuffer.AsSpan(0, outputCapacity),
                frameSize,
                false);
        }

        if (decodedFrames < 0)
        {
            throw new InvalidDataException($"The Opus decoder returned an invalid frame count: {decodedFrames}.");
        }

        return decodedFrames * _channels;
    }

    private int DecodeNextOggPacket()
    {
        if (_oggReader is null || !_oggReader.HasNextPacket)
        {
            return -1;
        }

        short[]? decoded = _oggReader.DecodeNextPacket();
        if (decoded is null)
        {
            if (!string.IsNullOrWhiteSpace(_oggReader.LastError))
            {
                throw new InvalidDataException($"Failed to decode Ogg Opus: {_oggReader.LastError}");
            }

            return -1;
        }

        EnsureCapacity(ref _decodeBuffer, decoded.Length);
        for (int i = 0; i < decoded.Length; i++)
        {
            _decodeBuffer[i] = decoded[i] / 32768f;
        }

        return decoded.Length;
    }

    private void ReadPacket(int packetLength)
    {
        int totalRead = 0;
        while (totalRead < packetLength)
        {
            int read = _reader!.Read(_packetBuffer, totalRead, packetLength - totalRead);
            if (read == 0)
            {
                throw new InvalidDataException("The length-prefixed Opus packet is truncated.");
            }

            totalRead += read;
        }
    }

    private static void EnsureCapacity<T>(ref T[] buffer, int requiredLength)
    {
        if (buffer.Length >= requiredLength)
        {
            return;
        }

        int newLength = buffer.Length == 0
            ? Math.Max(requiredLength, 256)
            : Math.Max(requiredLength, buffer.Length * 2);
        buffer = GC.AllocateUninitializedArray<T>(newLength);
    }

    private void InitializeOggReader()
    {
        // OpusOggReadStream 总会关闭传入流，因此用非拥有包装器保护调用方流，
        // 底层流最终是否关闭仍由 leaveOpen 决定。
        _oggReader = new OpusOggReadStream(_decoder, new NonDisposingStream(_stream));

        if (!_oggReader.HasNextPacket && !string.IsNullOrWhiteSpace(_oggReader.LastError))
        {
            string error = _oggReader.LastError;
            _oggReader.Close();
            _oggReader = null;
            throw new InvalidDataException($"Invalid Ogg Opus stream: {error}");
        }
    }

    private static bool HasOggSignature(Stream stream)
    {
        long position = stream.Position;
        Span<byte> signature = stackalloc byte[4];
        int bytesRead = 0;

        try
        {
            while (bytesRead < signature.Length)
            {
                int count = stream.Read(signature[bytesRead..]);
                if (count == 0)
                {
                    break;
                }

                bytesRead += count;
            }
        }
        finally
        {
            stream.Position = position;
        }

        return bytesRead == signature.Length && signature.SequenceEqual(OggSignature);
    }

    private static Stream OpenValidatedFile(string filePath, int sampleRate, int channels, int frameSizeMs)
    {
        ValidateParameters(sampleRate, channels, frameSizeMs);
        return File.OpenRead(filePath);
    }

    private static void ValidateParameters(int sampleRate, int channels, int frameSizeMs)
    {
        if (sampleRate is not (8000 or 12000 or 16000 or 24000 or 48000))
        {
            throw new ArgumentException(
                "Invalid sample rate. Opus supports: 8k, 12k, 16k, 24k, 48k Hz",
                nameof(sampleRate));
        }

        if (channels is < 1 or > 2)
        {
            throw new ArgumentException(
                "Invalid channel count. Opus supports 1 (mono) or 2 (stereo) channels",
                nameof(channels));
        }

        if (frameSizeMs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frameSizeMs), "Frame size must be positive.");
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
        {
            return;
        }

        if (!disposing)
        {
            _disposed = true;
            return;
        }

        try
        {
            _oggReader?.Close();
            _reader?.Dispose();
            (_decoder as IDisposable)?.Dispose();
            _packetBuffer = [];
            _shortDecodeBuffer = [];
            _decodeBuffer = [];
            _decodedSampleCount = 0;
            _remainingSamplesOffset = 0;
        }
        finally
        {
            if (!_leaveOpen)
            {
                _stream.Dispose();
            }

            _disposed = true;
        }
    }

    /// <summary>将 Dispose 屏蔽掉，其余操作全部转发到底层流。</summary>
    private sealed class NonDisposingStream(Stream inner) : Stream
    {
        public override bool CanRead => inner.CanRead;
        public override bool CanSeek => inner.CanSeek;
        public override bool CanWrite => inner.CanWrite;
        public override long Length => inner.Length;
        public override long Position { get => inner.Position; set => inner.Position = value; }
        public override void Flush() => inner.Flush();
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override int Read(Span<byte> buffer) => inner.Read(buffer);
        public override int ReadByte() => inner.ReadByte();
        public override long Seek(long offset, SeekOrigin origin) => inner.Seek(offset, origin);
        public override void SetLength(long value) => inner.SetLength(value);
        public override void Write(byte[] buffer, int offset, int count) => inner.Write(buffer, offset, count);
        public override void Write(ReadOnlySpan<byte> buffer) => inner.Write(buffer);
    }
}
