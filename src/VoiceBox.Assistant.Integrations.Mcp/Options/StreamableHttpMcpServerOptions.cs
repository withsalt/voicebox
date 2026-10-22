namespace VoiceBox.Assistant.Integrations.Mcp;

public sealed class StreamableHttpMcpServerOptions : McpToolServerOptions
{
    public Uri? Endpoint { get; set; }

    public IDictionary<string, string> Headers { get; } =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    internal override McpServerRegistration Freeze(string alias)
    {
        ValidateCommon(alias);
        if (Endpoint is null || !Endpoint.IsAbsoluteUri ||
            Endpoint.Scheme is not ("http" or "https"))
        {
            throw new ArgumentException("Endpoint must be an absolute HTTP(S) URI.", nameof(Endpoint));
        }

        return McpServerRegistration.StreamableHttp(
            alias,
            Endpoint,
            new Dictionary<string, string>(Headers, StringComparer.OrdinalIgnoreCase),
            this);
    }
}
