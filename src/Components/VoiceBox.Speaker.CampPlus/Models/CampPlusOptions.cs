using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Speaker.CampPlus.Models
{
    /// <summary>
    /// CAM++ 说话人验证服务的配置。
    /// 模型来源: ModelScope iic/speech_campplus_sv_zh-cn_16k-common (192 维 embedding, 16kHz)。
    /// </summary>
    public class CampPlusOptions : OnnxExecutionOptions
    {
        /// <summary>音频采样率 (Hz)。模型固定要求 16000Hz。</summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>声道数。仅支持 1 (单声道)。</summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>采样格式。仅支持 Float32 ([-1, 1])。</summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>ONNX intra-op 线程数。模型较小 (6.9M 参数), 默认 2 兼顾时延与 CPU 占用。</summary>
        public override int ThreadsNum { get; set; } = 2;

        /// <summary>
        /// 模型目录 (含 campplus.onnx 与 config.ini)。
        /// 为空时使用内置目录 runtimes/models/speaker/campplus (随包拷贝到输出目录)。
        /// </summary>
        public string ModelDir { get; set; } = string.Empty;

        /// <summary>
        /// 判定阈值 (余弦相似度)。null 时使用模型 config.ini 的官方值 (0.31, 整段验证标定)。
        /// 流式短窗得分分布更散, 实际部署建议按场景实测后调整。
        /// </summary>
        public float? Threshold { get; set; }

        /// <summary>流式检测的滑动窗口时长 (毫秒)。窗口越长得分越稳, 反应越慢。默认 1000ms。</summary>
        public int WindowMs { get; set; } = 1000;

        /// <summary>流式检测的窗口滑动间隔 (毫秒)。默认 250ms (每 250ms 产出一次结果)。</summary>
        public int HopMs { get; set; } = 250;

        /// <summary>
        /// 得分指数平滑系数 (0, 1]: 新窗口得分的权重, 1 表示不平滑。默认 0.6。
        /// </summary>
        public float ScoreSmoothing { get; set; } = 0.6f;

        /// <summary>
        /// 静音门: 窗口内音频 RMS ([-1,1] 幅度) 低于该值时跳过推理不产出结果
        /// (静音段的 embedding 无意义)。0 表示禁用。默认 0.005。
        /// </summary>
        public float MinWindowRms { get; set; } = 0.005f;

        /// <summary>
        /// 是否启用前缀窗提前打分 (early-exit): 首个完整窗就绪前, 对已积累的前缀音频
        /// 提前推理并产出 IsEarlyExit=true 的临时结果 (得分不参与平滑, 不影响完整窗结果)。
        /// 默认 false (保持整窗行为)。
        /// </summary>
        public bool EnableEarlyExit { get; set; }

        /// <summary>
        /// 前缀窗提前打分的最短前缀时长 (毫秒), 之后每 <see cref="HopMs"/> 再产出一次,
        /// 直到首个完整窗就绪。仅在 <see cref="EnableEarlyExit"/> 时生效;
        /// 必须在 [100, WindowMs) 内。默认 300ms。
        /// </summary>
        public int EarlyExitMinMs { get; set; } = 300;

        /// <summary>注册说话人所需的最短音频时长 (毫秒)。默认 1000ms, 建议实际提供 3~10s。</summary>
        public int MinEnrollMs { get; set; } = 1000;
    }
}
