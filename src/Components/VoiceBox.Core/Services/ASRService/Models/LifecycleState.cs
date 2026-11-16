using System;
using System.Collections.Generic;
using System.Text;

namespace VoiceBox.Core.Services.ASRService.Models
{
    public enum LifecycleState
    {
        Created,
        Starting,
        Active,
        Finishing,
        Cancelling,
        Released,
        Faulted,
    }
}
