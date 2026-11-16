using VoiceBox.Core.Services.DNRService;

namespace VoiceBox.DNR.Hush
{
    /// <summary>
    /// 基于 Hush 的实时数字降噪服务标识接口。
    /// 语义与 <see cref="IDigitalNoiseReductionService"/> 完全一致，仅用于按实现类型解析实例。
    /// </summary>
    public interface IHushDNRService : IDigitalNoiseReductionService
    {
    }
}
