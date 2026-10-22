using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Assistant.Modules.Speaker.Contracts;

namespace VoiceBox.Assistant.Modules.Wakeup.Contracts
{
    /// <summary>
    /// Composition-time factory for the optional wakeup gate. Speaker evidence is present only when
    /// <see cref="Api.Configuration.SpeakerConfirmationOptions.Enable"/> is on.
    /// </summary>
    internal interface IWakeupModuleFactory
    {
        /// <summary>Creates the gate and supplies the host logger for wake-word diagnostics.</summary>
        IWakeupModule Create(
            ISpeakerEvidenceProvider? speakerEvidence,
            TimeProvider timeProvider,
            ILogger? logger = null);
    }
}
