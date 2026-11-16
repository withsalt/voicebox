// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>


// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>


// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>


// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>


// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>


// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>


// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>


// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>

using VoiceBox.Audio.PortAudio.Backend.Enums;

namespace VoiceBox.Audio.PortAudio.Backend
{
    internal class PortAudioException : Exception
    {
        /// <summary>
        /// Error code (from the native PortAudio library).  Use `PortAudio.GetErrorText()` for some more details.
        /// </summary>
        public ErrorCode ErrorCode { get; private set; }

        /// <summary>
        /// Creates a new PortAudio error.
        /// </summary>
        public PortAudioException(ErrorCode ec) : base()
        {
            ErrorCode = ec;
        }

        /// <summary>
        /// Creates a new PortAudio error with a message attached.
        /// </summary>
        /// <param name="message">Message to send</param>
        public PortAudioException(ErrorCode ec, string message)
            : base(message)
        {
            ErrorCode = ec;
        }

        /// <summary>
        /// Creates a new PortAudio error with a message attached and an inner error.
        /// </summary>
        /// <param name="message">Message to send</param>
        /// <param name="inner">The exception that occured inside of this one</param>
        public PortAudioException(ErrorCode ec, string message, Exception inner)
            : base(message, inner)
        {
            ErrorCode = ec;
        }
    }
}
