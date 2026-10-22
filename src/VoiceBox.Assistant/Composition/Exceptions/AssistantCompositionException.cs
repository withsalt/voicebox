namespace VoiceBox.Assistant.Composition.Exceptions;

public class AssistantCompositionException : InvalidOperationException
{
    public string Code { get; set; }

    public AssistantCompositionException(string code, string message) : base(message)
    {
        this.Code = code;
    }
}
