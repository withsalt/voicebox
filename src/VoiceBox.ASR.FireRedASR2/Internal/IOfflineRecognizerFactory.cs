namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Factory for creating <see cref="OfflineRecognizer"/> instances. Inversion of control:
    /// the service receives the factory rather than constructing the recognizer directly so
    /// tests can swap in fakes and the model-loading concern stays out of the service.
    /// </summary>
    internal interface IOfflineRecognizerFactory
    {
        OfflineRecognizer Create(FireRed2ASROptions options);
    }

    internal sealed class FileSystemOfflineRecognizerFactory : IOfflineRecognizerFactory
    {
        public OfflineRecognizer Create(FireRed2ASROptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            if (string.IsNullOrWhiteSpace(options.ModelPath))
                throw new ArgumentException("ModelPath must be specified.", nameof(options));
            if (!Directory.Exists(options.ModelPath))
                throw new DirectoryNotFoundException($"Model directory not found: {options.ModelPath}");

            bool isInt8 = false;
            var files = Directory.GetFiles(options.ModelPath);
            if (files.Any(s=>s.Contains("decoder.int8.onnx", StringComparison.OrdinalIgnoreCase)) 
                && files.Any(s => s.Contains("encoder.int8.onnx", StringComparison.OrdinalIgnoreCase)))
            {
                isInt8 = true;
            }

            string suffix = isInt8 ? ".int8.onnx" : ".onnx";
            string mvn = Path.Combine(options.ModelPath, "am.mvn");
            string tokens = Path.Combine(options.ModelPath, "tokens.txt");
            string encoder = Path.Combine(options.ModelPath, "encoder" + suffix);
            string decoder = Path.Combine(options.ModelPath, "decoder" + suffix);
            string ctc = Path.Combine(options.ModelPath, "ctc" + suffix);

            return new OfflineRecognizer(
                encoderFilePath: encoder,
                decoderFilePath: decoder,
                mvnFilePath: mvn,
                tokensFilePath: tokens,
                ctcFilePath: ctc,
                onnxOptions: options);
        }
    }
}
