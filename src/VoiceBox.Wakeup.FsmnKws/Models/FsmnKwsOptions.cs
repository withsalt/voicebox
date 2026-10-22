using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Wakeup.FsmnKws.Models
{
    /// <summary>
    /// FsmnKws (基于 FunASR FSMN+CTC 字建模的流式唤醒词服务, 例如"小云小云"模型) 的配置。
    /// <para>
    /// 模型来源: ModelScope iic/speech_charctc_kws_phone-xiaoyun, 经
    /// scripts/kws-xiaoyun-onnx-export 导出为带 per-layer cache 的流式 ONNX。
    /// 唤醒词由模型在导出时确定, 运行期从模型目录的 kws_meta.json 读取
    /// (见 <see cref="FsmnKwsMetadata"/>), 不再由调用方手动配置。
    /// </para>
    /// </summary>
    public class FsmnKwsOptions : OnnxExecutionOptions
    {
        /// <summary>音频采样率 (Hz)。模型固定要求 16000Hz, 其它采样率请自行重采样。</summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>声道数。仅支持 1 (单声道)。</summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>采样格式。仅支持 Float32 ([-1, 1] 归一化振幅)。</summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>
        /// ONNX intra-op 线程数。本模型仅 ~750K 参数, 单线程即远超实时
        /// (多线程反而引入线程池调度/自旋开销), 故默认 1 而非基类的"自动 (半核数)"。
        /// </summary>
        public override int ThreadsNum { get; set; } = 1;

        /// <summary>流式 KWS ONNX 模型路径 (kws_streaming.onnx)。</summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// 可选: kws_meta.json 路径 (提供唤醒词、token id 序列及 CMVN)。
        /// 为空时默认取 <see cref="ModelPath"/> 所在目录下的 <see cref="FsmnKwsMetadata.FileName"/>;
        /// 文件缺失或缺少必要字段会在构造服务时报错。
        /// </summary>
        public string MetaFilePath { get; set; } = string.Empty;

        /// <summary>
        /// 全局触发阈值 (0~1]。命中分数 = 关键词各 token 峰值后验概率几何均值的 sqrt 形式
        /// (与 FunASR KwsCtcPrefixDecoder 一致)。默认 0.5。
        /// </summary>
        public float Threshold { get; set; } = 0.5f;

        /// <summary>CTC prefix beam search 每帧 token 候选数 (score beam)。默认 3。</summary>
        public int ScoreBeam { get; set; } = 3;

        /// <summary>CTC prefix beam search 保留的路径数 (path beam)。默认 20。</summary>
        public int PathBeam { get; set; } = 20;

        /// <summary>每帧 token 后验低于该值不进入搜索。默认 0.05。</summary>
        public float TokenProbFloor { get; set; } = 0.05f;

        /// <summary>关键词允许的最大发音时长 (秒), 超过该跨度的匹配丢弃。默认 3.0s。</summary>
        public float MaxKeywordSeconds { get; set; } = 3.0f;

        /// <summary>全局防抖时间 (ms)。命中后该时间内不再重复触发同一关键词。默认 800ms。</summary>
        public int DebounceMs { get; set; } = 800;
    }
}
