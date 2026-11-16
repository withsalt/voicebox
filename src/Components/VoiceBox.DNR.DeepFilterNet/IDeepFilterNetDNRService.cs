using VoiceBox.Core.Services.DNRService;

namespace VoiceBox.DNR.DeepFilterNet
{
    /// <summary>
    /// 基于 DeepFilterNet 的实时数字降噪服务标识接口。
    /// 语义与 <see cref="IDigitalNoiseReductionService"/> 完全一致，仅用于按实现类型解析实例。
    /// </summary>
    public interface IDeepFilterNetDNRService : IDigitalNoiseReductionService
    {
    }
}
