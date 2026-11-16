using VoiceBox.AEC.DFSMN.Models;

namespace VoiceBox.AEC.DFSMN.Internal
{
    internal static class DfsmnModelFiles
    {
        public static string ResolveModelPath(DfsmnAecOptions options)
        {
            if (!string.IsNullOrWhiteSpace(options.ModelPath))
            {
                if (!File.Exists(options.ModelPath))
                    throw new FileNotFoundException($"DFSMN AEC model not found: {options.ModelPath}", options.ModelPath);
                return options.ModelPath;
            }

            string path = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "aec", "dfsmn", "dfsmn_aec_mask_vad_stream.onnx");
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Built-in DFSMN AEC model not found at '{path}'. Set DfsmnAecOptions.ModelPath or copy the exported model to the runtime directory.",
                    path);
            }
            return path;
        }

        public static string ResolveMvnPath(DfsmnAecOptions options)
        {
            if (!string.IsNullOrWhiteSpace(options.MvnPath))
            {
                if (!File.Exists(options.MvnPath))
                    throw new FileNotFoundException($"DFSMN AEC MVN file not found: {options.MvnPath}", options.MvnPath);
                return options.MvnPath;
            }

            string path = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "aec", "dfsmn", "feature.DEY.mvn.txt");
            if (File.Exists(path)) return path;

            throw new FileNotFoundException(
                $"Built-in DFSMN AEC MVN file not found at '{path}'. Set DfsmnAecOptions.MvnPath or copy feature.DEY.mvn.txt to the runtime directory.",
                path);
        }
    }
}
