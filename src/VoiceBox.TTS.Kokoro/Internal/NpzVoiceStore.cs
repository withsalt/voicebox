using System.Collections.Concurrent;
using System.IO.Compression;
using System.Buffers.Binary;

namespace VoiceBox.TTS.Kokoro.Internal
{
    internal sealed class NpzVoiceStore
    {
        private readonly Dictionary<string, VoiceData> _voices;
        private readonly ConcurrentDictionary<(string Voice, int Row), (float[] Style, int[] Shape)> _styleCache = new();

        private NpzVoiceStore(Dictionary<string, VoiceData> voices)
        {
            _voices = voices;
        }

        public static NpzVoiceStore Load(string path)
        {
            var voices = new Dictionary<string, VoiceData>(StringComparer.Ordinal);

            using var stream = File.OpenRead(path);
            using var archive = new ZipArchive(stream, ZipArchiveMode.Read);

            foreach (var entry in archive.Entries)
            {
                string name = entry.Name;
                if (!name.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
                    continue;

                string voiceName = name[..^4];

                using var entryStream = entry.Open();
                var data = ReadNpyArray(entryStream);
                voices[voiceName] = data;
            }

            return new NpzVoiceStore(voices);
        }

        public (float[] Data, int[] StyleShape) GetVoiceStyle(string name, int tokenCount)
        {
            if (!_voices.TryGetValue(name, out var voice))
                throw new ArgumentException($"Voice '{name}' not found in voice store.");

            int firstDim = voice.Shape[0];
            int row = Math.Min(tokenCount, firstDim - 1);

            if (_styleCache.TryGetValue((name, row), out var cached))
                return cached;

            int sliceSize = voice.Data.Length / firstDim;
            float[] style = new float[sliceSize];
            Array.Copy(voice.Data, row * sliceSize, style, 0, sliceSize);

            int[] styleShape = voice.Shape.Length > 1
                ? voice.Shape[1..]
                : new[] { sliceSize };

            var entry = (style, styleShape);
            _styleCache.TryAdd((name, row), entry);
            return entry;
        }

        public IReadOnlyList<string> GetVoiceNames()
        {
            var names = _voices.Keys.ToList();
            names.Sort(StringComparer.Ordinal);
            return names;
        }

        public bool HasVoice(string name) => _voices.ContainsKey(name);

        public (float[] Data, int[] StyleShape) GetAnyVoiceStyle()
        {
            var first = _voices.Values.First();
            int sliceSize = first.Data.Length / first.Shape[0];
            float[] style = new float[sliceSize];
            Array.Copy(first.Data, 0, style, 0, sliceSize);
            int[] styleShape = first.Shape.Length > 1 ? first.Shape[1..] : [sliceSize];
            return (style, styleShape);
        }

        private static VoiceData ReadNpyArray(Stream stream)
        {
            using var reader = new BinaryReader(stream);

            byte[] magic = reader.ReadBytes(6);
            if (magic[0] != 0x93 || magic[1] != (byte)'N' || magic[2] != (byte)'U' ||
                magic[3] != (byte)'M' || magic[4] != (byte)'P' || magic[5] != (byte)'Y')
                throw new InvalidDataException("Invalid NPY magic number.");

            byte majorVersion = reader.ReadByte();
            byte minorVersion = reader.ReadByte();

            int headerLen;
            if (majorVersion == 1)
                headerLen = reader.ReadUInt16();
            else
                headerLen = (int)reader.ReadUInt32();

            byte[] headerBytes = reader.ReadBytes(headerLen);
            string header = System.Text.Encoding.ASCII.GetString(headerBytes).Trim('\n', ' ', '\0');

            var (dtype, fortranOrder, shape) = ParseNpyHeader(header);

            if (dtype != "<f4" && dtype != "float32")
                throw new NotSupportedException($"Unsupported dtype: {dtype}. Only float32 is supported.");
            if (fortranOrder)
                throw new NotSupportedException("Fortran-order arrays are not supported.");
            if (shape.Length < 2)
                throw new NotSupportedException($"Expected at least 2D array, got {shape.Length}D.");

            int totalElements = 1;
            foreach (int dim in shape) totalElements *= dim;

            float[] data = new float[totalElements];
            byte[] buffer = reader.ReadBytes(totalElements * 4);
            for (int i = 0; i < totalElements; i++)
            {
                data[i] = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(i * 4, 4));
            }

            return new VoiceData(data, shape);
        }

        private static (string Dtype, bool FortranOrder, int[] Shape) ParseNpyHeader(string header)
        {
            string dtype = "";
            bool fortranOrder = false;
            int[] shape = Array.Empty<int>();

            int descrIdx = header.IndexOf("'descr':", StringComparison.Ordinal);
            if (descrIdx >= 0)
            {
                int start = header.IndexOf('\'', descrIdx + 8) + 1;
                int end = header.IndexOf('\'', start);
                dtype = header[start..end];
            }

            int fortranIdx = header.IndexOf("'fortran_order':", StringComparison.Ordinal);
            if (fortranIdx >= 0)
            {
                fortranOrder = header.Contains("True", StringComparison.Ordinal);
            }

            int shapeIdx = header.IndexOf("'shape':", StringComparison.Ordinal);
            if (shapeIdx >= 0)
            {
                int start = header.IndexOf('(', shapeIdx) + 1;
                int end = header.IndexOf(')', start);
                string shapeStr = header[start..end];
                shape = shapeStr
                    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .Select(int.Parse)
                    .ToArray();
            }

            return (dtype, fortranOrder, shape);
        }

        private sealed record VoiceData(float[] Data, int[] Shape);
    }
}
