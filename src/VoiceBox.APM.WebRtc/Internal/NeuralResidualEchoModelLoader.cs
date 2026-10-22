using System.Security.Cryptography;

namespace VoiceBox.APM.WebRtc.Internal
{
    internal static class NeuralResidualEchoModelLoader
    {
        internal const string BundledModelRelativePath =
            "runtimes/models/aec/webrtc/v20260212/model.tflite";

        internal const string BundledModelSha256 =
            "3A18833EAEB08BFFFB88A588C10DB67885F246BA4794FD0F1609F2CCF6C30B77";

        public static byte[] Load(string? configuredPath)
        {
            bool isBundledModel = configuredPath is null;
            string path = isBundledModel
                ? RuntimeAssetLocator.FindRequired(
                    BundledModelRelativePath,
                    "Bundled WebRTC neural residual echo model")
                : ResolveConfiguredPath(configuredPath!);

            byte[] model = File.ReadAllBytes(path);
            if (model.Length == 0)
                throw new InvalidDataException($"WebRTC neural residual echo model is empty: '{path}'.");

            if (isBundledModel)
            {
                string actualHash = Convert.ToHexString(SHA256.HashData(model));
                if (!string.Equals(actualHash, BundledModelSha256, StringComparison.Ordinal))
                {
                    throw new InvalidDataException(
                        $"Bundled WebRTC neural residual echo model checksum mismatch: " +
                        $"actual={actualHash}, expected={BundledModelSha256}, path='{path}'.");
                }
            }

            return model;
        }

        private static string ResolveConfiguredPath(string configuredPath)
        {
            string path = Path.IsPathRooted(configuredPath)
                ? Path.GetFullPath(configuredPath)
                : Path.GetFullPath(configuredPath, AppContext.BaseDirectory);

            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Configured WebRTC neural residual echo model was not found: '{path}'.",
                    path);
            }

            return path;
        }
    }
}
