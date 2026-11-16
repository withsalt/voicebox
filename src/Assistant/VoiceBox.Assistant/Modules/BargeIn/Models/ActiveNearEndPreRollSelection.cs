namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal readonly record struct ActiveNearEndPreRollSelection(
    string Source,
    int Samples,
    long StreamEpoch,
    long DecisionStartSample,
    long DecisionEndSample,
    long SelectedStartSample,
    long SelectedEndSample)
{
    public static ActiveNearEndPreRollSelection Empty(string source)
        => new(source, 0, 0, 0, 0, 0, 0);
}
