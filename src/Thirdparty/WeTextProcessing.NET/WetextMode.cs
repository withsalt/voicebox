using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WeTextProcessing.NET
{
    /// <summary>
    /// Specifies the normalization direction for <see cref="WetextProcessor"/>.
    /// </summary>
    public enum WetextMode
    {
        /// <summary>
        /// Text Normalization: converts written form to spoken form
        /// (e.g. "2023年" → "二零二三年", "$1.5" → "one dollar and fifty cents").
        /// </summary>
        TextNormalization,

        /// <summary>
        /// Inverse Text Normalization: converts spoken form to written form
        /// (e.g. "二零二三年" → "2023年", "one dollar fifty" → "$1.50").
        /// </summary>
        InverseTextNormalization,
    }
}
