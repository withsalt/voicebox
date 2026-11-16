using Microsoft.Extensions.Logging.Abstractions;
using ModelContextProtocol;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace VoiceBox.Assistant.Conversation.Mcp.Tests;

/// <summary>
/// When invoked with --mcp-server, runs a stdio MCP server used by the integration tests.
/// Otherwise the process exits immediately; the xUnit runner never invokes this entry point.
/// </summary>
internal static class TestServerProgram
{
    public static async Task<int> Main(string[] args)
    {
        if (!args.Contains("--mcp-server", StringComparer.Ordinal))
            return 0;

        var options = new McpServerOptions
        {
            ServerInfo = new Implementation
            {
                Name = "voicebox-mcp-test",
                Version = "1.0.0",
            },
            ToolCollection = new McpServerPrimitiveCollection<McpServerTool>(StringComparer.Ordinal),
        };
        options.ToolCollection.Add(McpServerTool.Create(
            (Func<string, string>)(text => $"echo:{text}"),
            new McpServerToolCreateOptions
            {
                Name = "echo",
                Description = "Echo the input text.",
            }));
        options.ToolCollection.Add(McpServerTool.Create(
            (Func<CancellationToken, Task<string>>)(async cancellationToken =>
            {
                await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
                return "done";
            }),
            new McpServerToolCreateOptions
            {
                Name = "wait",
                Description = "Wait until the call is canceled.",
            }));

        await using var transport = new StdioServerTransport("voicebox-mcp-test", NullLoggerFactory.Instance);
        await using McpServer server = McpServer.Create(transport, options, NullLoggerFactory.Instance, serviceProvider: null);
        await server.RunAsync(CancellationToken.None);
        return 0;
    }
}
