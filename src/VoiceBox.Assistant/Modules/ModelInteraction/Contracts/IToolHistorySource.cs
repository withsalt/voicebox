namespace VoiceBox.Assistant.Modules.ModelInteraction.Contracts;

/// <summary>Binds tool audit outcomes and provider exchanges to one history owner.</summary>
internal interface IToolHistorySource
{
    void BindToolHistorySink(IToolHistorySink sink);
}
