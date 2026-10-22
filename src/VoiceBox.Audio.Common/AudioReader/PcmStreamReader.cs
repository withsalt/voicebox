using System.Buffers.Binary;
using System.Runtime.InteropServices;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.Common.AudioReader;

/// <summary>
/// 流式 PCM 音频读取器
/// </summary>
public sealed class PcmStreamReader : IAudioStreamReader
{
    private const int DEFAULT_BUFFER_SIZE = 32768;

    private readonly Stream _stream;
    private readonly byte[] _internalBuffer;
    private readonly int _bitsPerSample;
    private readonly int _bytesPerSample;
    private readonly SampleFormat _sampleFormat;
    private readonly Endianness _endianness;
    private readonly bool _normalize;
    private readonly bool _leaveOpen;
    private readonly float _normalizationFactor;

    // 用于跟踪内部缓冲区的状态
    private int _bufferOffset = 0;
    private int _bytesInBuffer = 0;

    private bool _disposed = false;

    public PcmStreamReader(string filePath, int bitsPerSample, SampleFormat sampleFormat = SampleFormat.Float32, Endianness endianness = Endianness.LittleEndian, bool normalize = true, int bufferSize = DEFAULT_BUFFER_SIZE)
        : this(OpenValidatedFile(filePath, bitsPerSample, sampleFormat), bitsPerSample, sampleFormat, endianness, normalize, bufferSize, false)
    {

    }


    public PcmStreamReader(Stream stream, int bitsPerSample, SampleFormat sampleFormat = SampleFormat.Float32, Endianness endianness = Endianness.LittleEndian, bool normalize = true, int bufferSize = DEFAULT_BUFFER_SIZE, bool leaveOpen = true)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        ValidateFormat(bitsPerSample, sampleFormat);

        _bitsPerSample = bitsPerSample;
        _sampleFormat = sampleFormat;
        _endianness = endianness;
        _normalize = normalize;
        _leaveOpen = leaveOpen;
        _bytesPerSample = bitsPerSample / 8;

        // 预计算归一化因子（使用乘法替代除法）
        _normalizationFactor = normalize ? 1f / (float)Math.Pow(2, _bitsPerSample - 1) : 1f;

        if (bufferSize < _bytesPerSample)
            bufferSize = _bytesPerSample;
        else
            bufferSize = (bufferSize / _bytesPerSample) * _bytesPerSample;

        _internalBuffer = new byte[bufferSize];
        _stream = stream;
    }

    private static Stream OpenValidatedFile(string filePath, int bitsPerSample, SampleFormat sampleFormat)
    {
        ValidateFormat(bitsPerSample, sampleFormat);
        return File.OpenRead(filePath);
    }

    private static void ValidateFormat(int bitsPerSample, SampleFormat sampleFormat)
    {
        if (bitsPerSample % 8 != 0 || bitsPerSample < 8 || bitsPerSample > 32)
            throw new ArgumentException("Unsupported bit depth. Must be a multiple of 8, between 8 and 32.", nameof(bitsPerSample));
        if (sampleFormat == SampleFormat.Float32 && bitsPerSample != 32)
            throw new ArgumentException("Float32 format requires 32 bits per sample.", nameof(sampleFormat));
    }

    /// <summary>
    /// 从当前位置读取最多 buffer.Length 个样本。
    /// </summary>
    public int ReadSamples(Span<float> buffer)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(PcmStreamReader));
        if (buffer.IsEmpty)
            return 0;

        int samplesCopied = 0;

        while (samplesCopied < buffer.Length)
        {
            if (_bufferOffset >= _bytesInBuffer)
            {
                _bufferOffset = 0;
                _bytesInBuffer = _stream.Read(_internalBuffer, 0, _internalBuffer.Length);

                if (_bytesInBuffer == 0)
                    break;
            }

            if (_bytesInBuffer - _bufferOffset < _bytesPerSample)
            {
                if (!TryRefillPartialSample())
                    break;

                continue;
            }

            // 计算可处理的样本数
            int samplesToProcess = Math.Min(
                buffer.Length - samplesCopied,
                (_bytesInBuffer - _bufferOffset) / _bytesPerSample
            );

            int bytesToProcess = samplesToProcess * _bytesPerSample;
            ConvertSamples(
                _internalBuffer.AsSpan(_bufferOffset, bytesToProcess),
                buffer.Slice(samplesCopied, samplesToProcess));
            _bufferOffset += bytesToProcess;
            samplesCopied += samplesToProcess;
        }

        return samplesCopied;
    }

    private bool TryRefillPartialSample()
    {
        int remainingBytes = _bytesInBuffer - _bufferOffset;
        if (remainingBytes <= 0)
            return false;

        _internalBuffer.AsSpan(_bufferOffset, remainingBytes).CopyTo(_internalBuffer);
        _bufferOffset = 0;
        _bytesInBuffer = remainingBytes;

        while (_bytesInBuffer < _bytesPerSample)
        {
            int bytesRead = _stream.Read(_internalBuffer, _bytesInBuffer, _internalBuffer.Length - _bytesInBuffer);
            if (bytesRead == 0)
                return false;

            _bytesInBuffer += bytesRead;
        }

        return true;
    }

    #region Conversion Helpers
    private void ConvertSamples(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if (_sampleFormat == SampleFormat.Float32)
        {
            ConvertFloat32(source, destination);
            return;
        }

        switch (_bytesPerSample)
        {
            case 1:
                for (int i = 0; i < destination.Length; i++)
                {
                    long value = (sbyte)source[i];
                    destination[i] = _normalize ? value * _normalizationFactor : value;
                }
                break;
            case 2:
                ConvertInt16(source, destination);
                break;
            case 3:
                ConvertInt24(source, destination);
                break;
            case 4:
                ConvertInt32(source, destination);
                break;
            default:
                throw new NotSupportedException($"Unsupported bytes per sample: {_bytesPerSample}");
        }
    }

    private void ConvertInt16(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if ((_endianness == Endianness.LittleEndian) == BitConverter.IsLittleEndian)
        {
            ReadOnlySpan<short> samples = MemoryMarshal.Cast<byte, short>(source);
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = _normalize ? samples[i] * _normalizationFactor : samples[i];
            }
            return;
        }

        for (int i = 0, offset = 0; i < destination.Length; i++, offset += sizeof(short))
        {
            short value = _endianness == Endianness.LittleEndian
                ? BinaryPrimitives.ReadInt16LittleEndian(source[offset..])
                : BinaryPrimitives.ReadInt16BigEndian(source[offset..]);
            destination[i] = _normalize ? value * _normalizationFactor : value;
        }
    }

    private void ConvertInt24(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0, offset = 0; i < destination.Length; i++, offset += 3)
        {
            int value = _endianness == Endianness.LittleEndian
                ? source[offset] | source[offset + 1] << 8 | source[offset + 2] << 16
                : source[offset] << 16 | source[offset + 1] << 8 | source[offset + 2];
            if ((value & 0x800000) != 0)
            {
                value |= unchecked((int)0xFF000000);
            }

            destination[i] = _normalize ? value * _normalizationFactor : value;
        }
    }

    private void ConvertInt32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if ((_endianness == Endianness.LittleEndian) == BitConverter.IsLittleEndian)
        {
            ReadOnlySpan<int> samples = MemoryMarshal.Cast<byte, int>(source);
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = _normalize ? samples[i] * _normalizationFactor : samples[i];
            }
            return;
        }

        for (int i = 0, offset = 0; i < destination.Length; i++, offset += sizeof(int))
        {
            int value = _endianness == Endianness.LittleEndian
                ? BinaryPrimitives.ReadInt32LittleEndian(source[offset..])
                : BinaryPrimitives.ReadInt32BigEndian(source[offset..]);
            destination[i] = _normalize ? value * _normalizationFactor : value;
        }
    }

    private void ConvertFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if ((_endianness == Endianness.LittleEndian) == BitConverter.IsLittleEndian)
        {
            MemoryMarshal.Cast<byte, float>(source).CopyTo(destination);
            return;
        }

        for (int i = 0, offset = 0; i < destination.Length; i++, offset += sizeof(float))
        {
            destination[i] = _endianness == Endianness.LittleEndian
                ? BinaryPrimitives.ReadSingleLittleEndian(source[offset..])
                : BinaryPrimitives.ReadSingleBigEndian(source[offset..]);
        }
    }
    #endregion

    #region IDisposable Implementation
    public void Dispose()
    {
        if (!_disposed)
        {
            if (!_leaveOpen)
            {
                _stream?.Dispose();
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    public void Reset()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(PcmStreamReader));

        _stream.Position = 0;
        _bufferOffset = 0;
        _bytesInBuffer = 0;
    }
    #endregion
}
