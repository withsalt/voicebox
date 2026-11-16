using System;
using System.Collections.Generic;
using System.Text;

namespace VoiceBox.DNR.Hush.Models
{
    internal static class HushModelExtensions
    {
        /// <summary>返回模型对应的目录名（位于 runtimes/models/dnr/hush/ 下）。</summary>
        public static string ToDirName(this HushModel model) => model switch
        {
            HushModel.Hush16k => "hush16k",
            _ => throw new ArgumentOutOfRangeException(nameof(model), model, "Unknown Hush model."),
        };
    }
}
