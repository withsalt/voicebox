namespace VoiceBox.APM.WebRtc.Internal
{
    /// <summary>Resolves assets copied with the assembly under the runtimes tree.</summary>
    internal static class RuntimeAssetLocator
    {
        public static string? TryFind(string relativePath)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(relativePath);

            foreach (string baseDirectory in EnumerateProbeBaseDirectories())
            {
                string candidate = Path.GetFullPath(Path.Combine(baseDirectory, relativePath));
                if (File.Exists(candidate))
                    return candidate;
            }

            return null;
        }

        public static string FindRequired(string relativePath, string assetDescription)
        {
            return TryFind(relativePath) ?? throw new FileNotFoundException(
                $"{assetDescription} was not found at '{relativePath}' in any probed directory " +
                "(AppContext.BaseDirectory, ./bin, assembly location).",
                relativePath);
        }

        private static IEnumerable<string> EnumerateProbeBaseDirectories()
        {
            string baseDirectory = AppContext.BaseDirectory;
            if (!string.IsNullOrEmpty(baseDirectory))
            {
                yield return baseDirectory;
                yield return Path.Combine(baseDirectory, "bin");
            }

            string? assemblyDirectory = Path.GetDirectoryName(typeof(RuntimeAssetLocator).Assembly.Location);
            if (!string.IsNullOrEmpty(assemblyDirectory) &&
                !string.Equals(assemblyDirectory, baseDirectory, StringComparison.OrdinalIgnoreCase))
            {
                yield return assemblyDirectory;
            }
        }
    }
}
