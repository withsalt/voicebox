using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WeTextProcessing.NET
{
    /// <summary>
    /// Specifies the language for <see cref="WetextProcessor"/>.
    /// </summary>
    public enum WetextLanguage
    {
        /// <summary>
        /// Chinese. Uses <c>zh_*.fst</c> model files.
        /// </summary>
        Chinese,

        /// <summary>
        /// English. Uses <c>en_*.fst</c> model files.
        /// </summary>
        English,
    }
}
