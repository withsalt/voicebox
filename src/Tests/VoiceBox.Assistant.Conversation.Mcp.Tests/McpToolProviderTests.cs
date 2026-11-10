using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using Xunit;

namespace VoiceBox.Assistant.Conversation.Mcp.Tests;

public sealed class McpToolProviderTests
{
    private static string DotnetHost => Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ?? "dotnet";

    private static string TestAssemblyPath => typeof(McpToolProviderTests).Assembly.Location;

    private static McpToolProvider CreateStdioProvider(
        Action<StdioMcpServerOptions>? configure = null,
        Action<StdioMcpServerOptions>? capture = null)
        => McpToolProvider.CreateStdio("test", options =>
        {
            capture?.Invoke(options);
            options.Command = DotnetHost;
            options.Arguments.Add("exec");
            options.Arguments.Add(TestAssemblyPath);
            options.Arguments.Add("--mcp-server");
            options.WorkingDirectory = AppContext.BaseDirectory;
            options.ConnectTimeout = TimeSpan.FromSeconds(15);
            options.RequestTimeout = TimeSpan.FromSeconds(15);
            configure?.Invoke(options);
        });

    private static ConversationToolCall EchoCall(string text)
        => new(Guid.NewGuid().ToString("N"), new ConversationToolId("test", "echo"), $$"""{"text":"{{text}}"}""");

    [Fact]
    public async Task ProviderCanBeCreatedWithoutConnectingAndFrozenConfigurationSurvivesMutation()
    {
        StdioMcpServerOptions? captured = null;
        McpToolProvider provider = CreateStdioProvider(options => options.AllowedTools.Add("echo"), options => captured = options);

        Assert.Equal(["test"], provider.ServerAliases);
        Assert.Empty(provider.GetType().GetConstructors(BindingFlags.Public | BindingFlags.Instance));

        // The provider froze its own configuration; mutating the object passed to the creation
        // callback must not change what discovery exposes.
        Assert.NotNull(captured);
        captured!.AllowedTools.Add("wait");
        captured.RequestTimeout = TimeSpan.FromMilliseconds(1);

        await provider.StartAsync();
        try
        {
            IReadOnlyList<ConversationToolDefinition> tools = await provider.GetToolsAsync();
            Assert.Equal(["echo"], tools.Select(tool => tool.Id.ToolName));
            Assert.All(tools, tool => Assert.Equal("test", tool.Id.ServerAlias));
        }
        finally
        {
            await provider.StopAsync();
            await provider.DisposeAsync();
        }
    }

    [Fact]
    public async Task ProviderDiscoversInvokesStopsRestartsAndDisposes()
    {
        await using McpToolProvider provider = CreateStdioProvider(options => options.AllowedTools.Add("echo"));

        await provider.StartAsync();
        IReadOnlyList<ConversationToolDefinition> tools = await provider.GetToolsAsync();
        ConversationToolDefinition echo = Assert.Single(tools, tool => tool.Id.ToolName == "echo");
        Assert.Contains("Echo", echo.Descriptor.Description, StringComparison.Ordinal);

        ConversationToolResult first = await provider.InvokeAsync(EchoCall("one"));
        Assert.Equal(ConversationToolResultKind.Success, first.Kind);
        Assert.Contains("echo:one", first.ValueJson, StringComparison.Ordinal);
        await provider.StopAsync();

        await provider.StartAsync();
        try
        {
            ConversationToolResult second = await provider.InvokeAsync(EchoCall("two"));
            Assert.Equal(ConversationToolResultKind.Success, second.Kind);
            Assert.Contains("echo:two", second.ValueJson, StringComparison.Ordinal);
        }
        finally
        {
            await provider.StopAsync();
        }
    }

    [Fact]
    public async Task InvocationCancellationPropagates()
    {
        await using McpToolProvider provider = CreateStdioProvider(options => options.AllowedTools.Add("wait"));
        await provider.StartAsync();
        try
        {
            var call = new ConversationToolCall(Guid.NewGuid().ToString("N"), new ConversationToolId("test", "wait"), "{}");
            using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));

            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await provider.InvokeAsync(call, cancellation.Token));
        }
        finally
        {
            await provider.StopAsync();
        }
    }

    [Theory]
    [InlineData(ConversationToolRisk.ReadOnly, ConversationToolResultKind.TimedOut, false)]
    [InlineData(ConversationToolRisk.Sensitive, ConversationToolResultKind.TimedOut, false)]
    [InlineData(ConversationToolRisk.SideEffecting, ConversationToolResultKind.UnknownOutcome, false)]
    [InlineData(ConversationToolRisk.Destructive, ConversationToolResultKind.UnknownOutcome, false)]
    [InlineData(ConversationToolRisk.Privileged, ConversationToolResultKind.UnknownOutcome, false)]
    [InlineData(ConversationToolRisk.ReadOnly, ConversationToolResultKind.TimedOut, true)]
    [InlineData(ConversationToolRisk.SideEffecting, ConversationToolResultKind.UnknownOutcome, true)]
    public async Task StandaloneInvocationHonorsRequestTimeoutAndReleasesConcurrencySlot(
        ConversationToolRisk risk,
        ConversationToolResultKind expectedKind,
        bool useMcpTasks)
    {
        await using McpToolProvider provider = CreateStdioProvider(options =>
        {
            options.RequestTimeout = TimeSpan.FromSeconds(1);
            options.MaxConcurrentCalls = 1;
            options.DefaultRisk = risk;
            options.UseMcpTasks = useMcpTasks;
        });
        await provider.StartAsync();
        var call = new ConversationToolCall("timed-call", new ConversationToolId("test", "wait"), "{}");
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(10));

        ConversationToolResult result = await provider.InvokeAsync(call, cancellation.Token);

        Assert.False(cancellation.IsCancellationRequested);
        Assert.Equal(call.CallId, result.CallId);
        Assert.Equal(expectedKind, result.Kind);
        Assert.Equal("timeout", result.ErrorCode);

        ConversationToolResult next = await provider.InvokeAsync(EchoCall("after-timeout"), cancellation.Token);
        Assert.Equal(ConversationToolResultKind.Success, next.Kind);
        Assert.Contains("echo:after-timeout", next.ValueJson, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CancelingAQueuedInvocationPreservesTheActiveConcurrencySlot()
    {
        await using McpToolProvider provider = CreateStdioProvider(options => options.MaxConcurrentCalls = 1);
        await provider.StartAsync();
        using var activeCancellation = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var call = new ConversationToolCall("active", new ConversationToolId("test", "wait"), "{}");
        Task<ConversationToolResult> active = provider.InvokeAsync(call, activeCancellation.Token).AsTask();
        try
        {
            using var queuedCancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
                await provider.InvokeAsync(EchoCall("queued"), queuedCancellation.Token));
            Assert.False(active.IsCompleted);
        }
        finally
        {
            activeCancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => active);
        }

        ConversationToolResult next = await provider.InvokeAsync(EchoCall("after-cancel"));
        Assert.Equal(ConversationToolResultKind.Success, next.Kind);
    }

    [Fact]
    public async Task DiscoveryAndInvocationRequireAStartedProvider()
    {
        await using McpToolProvider provider = CreateStdioProvider(options => options.AllowedTools.Add("echo"));

        Assert.Throws<InvalidOperationException>(() => provider.GetToolsAsync());
        var call = new ConversationToolCall("call-1", new ConversationToolId("test", "echo"), """{"text":"x"}""");
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await provider.InvokeAsync(call));
    }

    [Fact]
    public async Task StartupFailureCleansUpAndDisposalCompletes()
    {
        McpToolProvider provider = McpToolProvider.CreateStdio("broken", options =>
        {
            options.Command = "voicebox-command-that-does-not-exist";
            options.ConnectTimeout = TimeSpan.FromMilliseconds(200);
        });

        await Assert.ThrowsAnyAsync<Exception>(() => provider.StartAsync());
        await provider.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public void CreationValidatesConfigurationWithoutStartingProcesses()
    {
        Assert.Throws<ArgumentException>(() => McpToolProvider.CreateStdio("bad alias!", options => options.Command = "echo"));
        Assert.Throws<ArgumentException>(() =>
            McpToolProvider.CreateStdio("no-command", _ => { }));
        Assert.Throws<ArgumentException>(() =>
            McpToolProvider.CreateStreamableHttp("no-endpoint", _ => { }));
        Assert.Throws<ArgumentNullException>(() => McpToolProvider.CreateStdio("null-config", null!));
    }

    [Fact]
    public async Task AProviderCanBackACustomSessionWithoutTheDefaultConversation()
    {
        await using McpToolProvider provider = CreateStdioProvider(options => options.AllowedTools.Add("echo"));
        await provider.StartAsync();
        try
        {
            await using var session = new ProviderBackedSession(provider);
            var events = new List<ConversationEvent>();
            await foreach (ConversationEvent item in session.StreamAsync(new ConversationRequest(Guid.NewGuid(), "session")))
            {
                events.Add(item);
            }

            Assert.Contains("echo:session", Assert.IsType<ConversationCompleted>(events[^1]).Text, StringComparison.Ordinal);
        }
        finally
        {
            await provider.StopAsync();
        }
    }

    [Fact]
    public void McpProjectDependsOnlyOnAbstractionsAndTheMcpSdk()
    {
        Assembly mcp = typeof(McpToolProvider).Assembly;
        Assert.Contains(mcp.GetReferencedAssemblies(), reference => reference.Name == "VoiceBox.Assistant.Conversation.Abstractions");
        Assert.DoesNotContain(
            mcp.GetReferencedAssemblies(),
            reference => reference.Name is
                "VoiceBox.Assistant.Conversation" or
                "VoiceBox.Assistant" or
                "VoiceBox.Core");

        string project = ReadProjectFile("src", "VoiceBox.Assistant.Conversation.Mcp", "VoiceBox.Assistant.Conversation.Mcp.csproj");
        string[] references = Regex.Matches(project, "ProjectReference Include=\"([^\"]+)\"")
            .Select(match => match.Groups[1].Value.Replace('\\', '/'))
            .ToArray();
        string reference = Assert.Single(references);
        Assert.EndsWith(
            "VoiceBox.Assistant.Conversation.Abstractions/VoiceBox.Assistant.Conversation.Abstractions.csproj",
            reference,
            StringComparison.Ordinal);
    }

    [Fact]
    public void TheTestProjectUsesOnlyPublicMcpAndAbstractionsProductReferences()
    {
        string project = ReadProjectFile(
            "src",
            "Tests",
            "VoiceBox.Assistant.Conversation.Mcp.Tests",
            "VoiceBox.Assistant.Conversation.Mcp.Tests.csproj");
        string[] references = Regex.Matches(project, "ProjectReference Include=\"([^\"]+)\"")
            .Select(match => match.Groups[1].Value.Replace('\\', '/'))
            .ToArray();

        Assert.Equal(2, references.Length);
        Assert.Contains(
            references,
            value => value.EndsWith(
                "VoiceBox.Assistant.Conversation.Abstractions/VoiceBox.Assistant.Conversation.Abstractions.csproj",
                StringComparison.Ordinal));
        Assert.Contains(
            references,
            value => value.EndsWith(
                "VoiceBox.Assistant.Conversation.Mcp/VoiceBox.Assistant.Conversation.Mcp.csproj",
                StringComparison.Ordinal));
        Assert.DoesNotContain(references, value => value.Contains("VoiceBox.Assistant.Conversation/VoiceBox", StringComparison.Ordinal));
        Assert.DoesNotContain(references, value => value.Contains("VoiceBox.Assistant/VoiceBox", StringComparison.Ordinal));
    }

    private static string ReadProjectFile(params string[] segments)
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            string candidate = Path.Combine([current.FullName, .. segments]);
            if (File.Exists(candidate))
                return File.ReadAllText(candidate);
            current = current.Parent;
        }

        throw new DirectoryNotFoundException("The requested project file was not found.");
    }

    private sealed class ProviderBackedSession(IConversationToolProvider provider) : IConversationSession
    {
        public string ConversationId => "provider-backed";

        public async IAsyncEnumerable<ConversationEvent> StreamAsync(
            ConversationRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            IReadOnlyList<ConversationToolDefinition> tools = await provider.GetToolsAsync(cancellationToken);
            ConversationToolDefinition echo = Assert.Single(tools, tool => tool.Id.ToolName == "echo");
            var call = new ConversationToolCall(request.RequestId.ToString("N"), echo.Id, $$"""{"text":"{{request.Text}}"}""");
            ConversationToolResult result = await provider.InvokeAsync(call, cancellationToken);
            string text = result.ValueJson ?? string.Empty;

            yield return new ConversationTextDelta(text);
            yield return new ConversationCompleted(text);
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
