using System;
using System.Collections.Generic;
using System.Text;

namespace VoiceBox.DNR.DPDFNet.Models
{
    /// <summary>
    /// <see cref="DPDFNetOptions.AttenuationLimitDb"/> 的施加方式。
    /// </summary>
    public enum DPDFNetAttenuationMode
    {
        /// <summary>
        /// 复数谱线性混合 (默认, 历史行为): <c>out = alpha·noisy + (1-alpha)·enhanced</c>。
        /// 实现简单, 但 noisy 与 enhanced 相位不一致时, 中等混合比 (alpha≈0.5) 会产生
        /// 相位干涉/梳状伪影, 对下游 ASR 反而不利 (实测 alpha≈0.5 即 AttenuationLimitDb≈6 处最差)。
        /// </summary>
        ComplexBlend = 0,

        /// <summary>
        /// 幅度域增益下限: 取模型隐含的每频点增益 <c>g=|enhanced|/|noisy|</c>, 下限钳到
        /// <c>alpha</c> (= 10^(-AttenuationLimitDb/20)), 再把该实增益作用到<b>原始 noisy 谱并保留其相位</b>。
        /// 不引入相位干涉, 因此可在保留更强降噪的同时显著降低对 ASR 的损伤。
        /// <see cref="DPDFNetOptions.AttenuationLimitDb"/> 为 null 时下限取 0 (纯幅度增强, 仍保留 noisy 相位)。
        /// </summary>
        MagnitudeFloor = 1,
    }
}
