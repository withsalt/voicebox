namespace VoiceBox.DNR.DPDFNet.Models
{
    /// <summary>
    /// DPDFNet 16 kHz 降噪模型档位。模型越大，降噪能力越强，但推理开销越高。
    /// 四个模型共享相同的输入/输出张量结构 (spec[1,1,161,2] + state)，
    /// 仅内部状态尺寸与网络容量不同，可直接互换。
    /// </summary>
    public enum DPDFNetModel
    {
        /// <summary>最轻量基线模型 (state_size=38256)。延迟最低，降噪强度最弱。</summary>
        Baseline = 0,

        /// <summary>dpdfnet2 (state_size=45424)。默认档位，速度与效果平衡。</summary>
        DpDfNet2 = 1,

        /// <summary>dpdfnet4 (state_size=52592)。更强降噪。</summary>
        DpDfNet4 = 2,

        /// <summary>dpdfnet8 (state_size=66928)。最强降噪，开销最高。</summary>
        DpDfNet8 = 3,
    }
}
