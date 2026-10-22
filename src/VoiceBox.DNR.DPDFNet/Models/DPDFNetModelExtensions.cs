using System;
using System.Collections.Generic;
using System.Text;

namespace VoiceBox.DNR.DPDFNet.Models
{
    internal static class DPDFNetModelExtensions
    {
        /// <summary>返回模型对应的 onnx 文件名 (不含目录)。</summary>
        public static string ToFileName(this DPDFNetModel model) => model switch
        {
            DPDFNetModel.Baseline => "baseline.onnx",
            DPDFNetModel.DpDfNet2 => "dpdfnet2.onnx",
            DPDFNetModel.DpDfNet4 => "dpdfnet4.onnx",
            DPDFNetModel.DpDfNet8 => "dpdfnet8.onnx",
            _ => throw new ArgumentOutOfRangeException(nameof(model), model, "Unknown DPDFNet model."),
        };
    }
}
