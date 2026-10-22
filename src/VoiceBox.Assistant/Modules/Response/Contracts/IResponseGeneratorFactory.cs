namespace VoiceBox.Assistant.Modules.Response.Contracts;

/// <summary>
/// Composition-time adapter boundary for a response provider. Creating the generator is delayed
/// until an assistant instance is being assembled so a validated composition remains inert.
/// </summary>
internal interface IResponseGeneratorFactory
{
    IResponseGenerator Create();
}
