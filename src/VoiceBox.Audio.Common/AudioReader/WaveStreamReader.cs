using System.Runtime.InteropServices;
using System.Text;
using VoiceBox.Audio.Common.AudioReader.Chunks;

namespace VoiceBox.Audio.Common.AudioReader;

/// <summary>
/// A class for reading WAVE audio data from a stream.
/// This version supports reading the audio data in user-defined chunks.
/// </summary>
public class WaveStreamReader : IAudioStreamReader
{
    private readonly Stream _stream;
    private readonly BinaryReader _reader;
    private readonly long _dataChunkPosition;
    private readonly uint _dataChunkSize;

    private long _dataBytesRead; // 已从数据块中读取的字节数
    private byte[] _sourceBuffer = [];
    private bool _disposed;

    /// <summary>
    /// The main riff chunk
    /// </summary>
    public RiffChunk MainChunk { get; } = new();

    /// <summary>
    /// The fact chunk
    /// </summary>
    public FactChunk? FactChunk { get; private set; }

    /// <summary>
    /// The format chunk
    /// </summary>
    public FmtChunk FormatChunk { get; private set; } = new();

    /// <summary>
    /// Gets the number of bytes remaining in the data chunk.
    /// </summary>
    public long RemainingBytes => _dataChunkSize - _dataBytesRead;

    public WaveStreamReader(string filePath) : this(File.OpenRead(filePath), false)
    {
    }

    public WaveStreamReader(Stream stream, bool leaveOpen = true)
    {
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen);

        try
        {
            ParseHeaders();

            // Initialize position
            _dataBytesRead = 0;
            _dataChunkPosition = GetAndUpdateDataChunkPosition();
            _dataChunkSize = GetChunkSize("data");
        }
        catch
        {
            _reader.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The main method to read a specified number of samples from the stream.
    /// </summary>
    /// <param name="buffer">The destination buffer to fill with samples.</param>
    /// <returns>The number of samples written to <paramref name="buffer"/>.</returns>
    public int ReadSamples(Span<float> buffer)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(WaveStreamReader));

        if (buffer.IsEmpty) return 0;

        int bytesPerSample = FormatChunk.BitsPerSample / 8;
        if (bytesPerSample == 0) return 0;

        long remainingSamples = RemainingBytes / bytesPerSample;
        if (remainingSamples == 0) return 0;

        int samplesToRead = (int)Math.Min(buffer.Length, remainingSamples);
        int bytesToRead = samplesToRead * bytesPerSample;

        EnsureSourceBufferCapacity(bytesToRead);
        int bytesRead = ReadSourceBytes(bytesToRead);

        if (bytesRead < bytesToRead)
            samplesToRead = bytesRead / bytesPerSample;

        if (samplesToRead == 0)
            return 0;

        ConvertBufferToFloat(_sourceBuffer, buffer, samplesToRead);

        _dataBytesRead += bytesRead;
        return samplesToRead;
    }

    private void EnsureSourceBufferCapacity(int requiredLength)
    {
        if (_sourceBuffer.Length >= requiredLength)
            return;

        int newLength = _sourceBuffer.Length == 0
            ? requiredLength
            : Math.Max(requiredLength, _sourceBuffer.Length * 2);
        _sourceBuffer = GC.AllocateUninitializedArray<byte>(newLength);
    }

    private int ReadSourceBytes(int bytesToRead)
    {
        int totalRead = 0;
        while (totalRead < bytesToRead)
        {
            int read = _stream.Read(_sourceBuffer, totalRead, bytesToRead - totalRead);
            if (read == 0)
                break;

            totalRead += read;
        }

        return totalRead;
    }

    /// <summary>
    /// Resets the reader's position to the beginning of the audio data.
    /// </summary>
    public void Reset()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(WaveStreamReader));

        // Resetting requires seeking.
        _stream.Position = _dataChunkPosition;
        _dataBytesRead = 0;
    }

    private void ConvertBufferToFloat(byte[] sourceBuffer, Span<float> destinationBuffer, int samplesToConvert)
    {
        int bytesPerSample = FormatChunk.BitsPerSample / 8;

        switch (FormatChunk.AudioFormat)
        {
            case Format.IeeeFloat when FormatChunk.BitsPerSample == 32:
                MemoryMarshal.Cast<byte, float>(sourceBuffer.AsSpan(0, samplesToConvert * 4))
                    .CopyTo(destinationBuffer[..samplesToConvert]);
                break;

            case Format.Uncompressed when FormatChunk.BitsPerSample == 16:
                // 使用 MemoryMarshal 进行高效类型转换
                var int16Span = MemoryMarshal.Cast<byte, short>(sourceBuffer.AsSpan(0, samplesToConvert * 2));
                for (int i = 0; i < samplesToConvert; i++)
                {
                    destinationBuffer[i] = int16Span[i] / 32768f;
                }
                break;

            case Format.Uncompressed when FormatChunk.BitsPerSample == 24:
                for (int i = 0; i < samplesToConvert; i++)
                {
                    int offset = i * 3;
                    // 符号位在第三个字节
                    int sample = sourceBuffer[offset] | sourceBuffer[offset + 1] << 8 | (sbyte)sourceBuffer[offset + 2] << 16;
                    destinationBuffer[i] = sample / 8388608f;
                }
                break;

            case Format.Uncompressed when FormatChunk.BitsPerSample == 8:
                for (int i = 0; i < samplesToConvert; i++)
                {
                    destinationBuffer[i] = (sourceBuffer[i] - 128) / 128f;
                }
                break;

            default:
                throw new NotSupportedException($"Unsupported bit depth/format combination: {FormatChunk.BitsPerSample}-bit {FormatChunk.AudioFormat}");
        }
    }

    #region Header Parsing and Helpers
    private void ParseHeaders()
    {
        if (!MainChunk.Parse(_reader))
            throw new InvalidDataException("Input stream is not a valid WAVE file.");

        while (_reader.BaseStream.Position < _reader.BaseStream.Length)
        {
            long chunkPosition = _reader.BaseStream.Position;
            Chunk chunkInfo = Chunk.PeekInfo(_reader);
            if (chunkInfo.Id == "fmt ")
            {
                FormatChunk = new FmtChunk();
                FormatChunk.Parse(_reader);
                MoveToNextChunk(chunkPosition, chunkInfo.Size);
            }
            else if (chunkInfo.Id == "fact")
            {
                FactChunk = new FactChunk();
                FactChunk.Parse(_reader);
                MoveToNextChunk(chunkPosition, chunkInfo.Size);
            }
            else if (chunkInfo.Id == "data")
            {
                // We've found all necessary metadata, no need to parse further.
                return;
            }
            else
            {
                _reader.BaseStream.Seek(8 + GetPaddedChunkSize(chunkInfo.Size), SeekOrigin.Current);
            }
        }
    }

    private void MoveToNextChunk(long chunkPosition, uint chunkSize)
    {
        long nextChunkPosition = checked(chunkPosition + 8 + GetPaddedChunkSize(chunkSize));
        if (nextChunkPosition > _reader.BaseStream.Length)
            throw new InvalidDataException($"WAVE chunk at position {chunkPosition} extends beyond the input stream.");

        _reader.BaseStream.Position = nextChunkPosition;
    }

    private long GetAndUpdateDataChunkPosition()
    {
        _stream.Position = 12; // Skip RIFF header
        while (_stream.Position < _stream.Length)
        {
            var chunkId = new string(_reader.ReadChars(4));
            var chunkSize = _reader.ReadUInt32();
            if (chunkId == "data")
            {
                return _stream.Position;
            }
            _stream.Seek(GetPaddedChunkSize(chunkSize), SeekOrigin.Current);
        }
        throw new InvalidDataException("'data' chunk not found.");
    }

    private uint GetChunkSize(string chunkId)
    {
        long originalPosition = _stream.Position;
        _stream.Position = 12; // Skip RIFF header
        try
        {
            while (_stream.Position < _stream.Length)
            {
                var id = new string(_reader.ReadChars(4));
                var size = _reader.ReadUInt32();
                if (id == chunkId)
                {
                    return size;
                }
                _stream.Seek(GetPaddedChunkSize(size), SeekOrigin.Current);
            }
        }
        finally
        {
            _stream.Position = originalPosition;
        }
        throw new InvalidDataException($"'{chunkId}' chunk not found.");
    }

    private static long GetPaddedChunkSize(uint chunkSize)
    {
        return chunkSize + (chunkSize & 1u);
    }
    #endregion

    #region IDisposable
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            _reader?.Dispose();
            _sourceBuffer = [];
        }

        _disposed = true;
    }
    #endregion
}
