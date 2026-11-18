using VoiceBox.Core.Onnx;
using VoiceBox.DNR.GRENet.Models;

namespace VoiceBox.DNR.GRENet.Internal;

internal static class GRENetOnnxRunnerFactory
{
    public static IGRENetOnnxRunner Create(string modelPath, GRENetOptions options)
        => options.ExecutionProvider switch
        {
            OnnxExecutionProvider.Cpu => new GRENetCpuOnnxRunner(modelPath, options),
            OnnxExecutionProvider.Cuda => new GRENetCudaOnnxRunner(modelPath, options),
            _ => throw new NotSupportedException(
                $"GRENetDNRService supports CPU and CUDA execution providers only, got {options.ExecutionProvider}."),
        };
}
