namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

internal readonly record struct RenderDelayHint(int DelayMilliseconds, long TimestampTicks, bool IsStable)
{
    public bool IsValid => DelayMilliseconds >= 0 && TimestampTicks >= 0;
}
