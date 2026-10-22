using System.Globalization;
using System.Text.RegularExpressions;

namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class MvnTransform
    {
        private readonly float[] _shift;
        private readonly float[] _scale;

        public int Dimension => _shift.Length;

        private MvnTransform(float[] shift, float[] scale)
        {
            _shift = shift;
            _scale = scale;
        }

        public static MvnTransform Identity(int dimension)
        {
            if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
            var shift = new float[dimension];
            var scale = new float[dimension];
            Array.Fill(scale, 1f);
            return new MvnTransform(shift, scale);
        }

        public static MvnTransform Load(string path, int expectedDimension)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"MVN transform not found: {path}", path);

            string text = File.ReadAllText(path);
            var arrays = Regex.Matches(text, @"\[(.*?)\]", RegexOptions.Singleline)
                .Select(m => ParseVector(m.Groups[1].Value))
                .Where(a => a.Length == expectedDimension)
                .ToArray();

            if (arrays.Length < 2)
                throw new InvalidDataException($"MVN transform '{path}' must contain AddShift and Rescale vectors.");

            float[] shift = arrays[0];
            float[] scale = arrays[1];
            if (shift.Length != expectedDimension || scale.Length != expectedDimension)
                throw new InvalidDataException(
                    $"MVN transform '{path}' dimension mismatch. Expected {expectedDimension}, got shift={shift.Length}, scale={scale.Length}.");

            return new MvnTransform(shift, scale);
        }

        public void Apply(Span<float> features)
        {
            if (features.Length != _shift.Length)
                throw new ArgumentException($"feature length must be {_shift.Length}, got {features.Length}.", nameof(features));

            for (int i = 0; i < features.Length; i++)
                features[i] = (features[i] + _shift[i]) * _scale[i];
        }

        private static float[] ParseVector(string body)
        {
            return body.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => float.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture))
                .ToArray();
        }
    }
}
