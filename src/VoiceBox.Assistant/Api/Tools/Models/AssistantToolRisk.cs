using System;
using System.Collections.Generic;
using System.Text;

namespace VoiceBox.Assistant.Api.Tools.Models
{
    public enum AssistantToolRisk
    {
        ReadOnly,
        Sensitive,
        SideEffecting,
        Destructive,
        Privileged,
    }
}
