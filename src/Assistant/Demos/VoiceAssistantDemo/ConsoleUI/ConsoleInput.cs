namespace VoiceAssistantDemo.ConsoleUI;

/// <summary>等待按键时不占用显示锁，并允许取消或通过重定向输入提供命令。</summary>
internal sealed class ConsoleInput
{
    private readonly TextReader _input;
    private readonly bool _interactive;

    public ConsoleInput(TextReader input, bool interactive)
    {
        ArgumentNullException.ThrowIfNull(input);
        _input = input;
        _interactive = interactive;
    }

    public async Task<ConsoleKey> ReadKeyAsync(CancellationToken cancellationToken = default)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (_interactive)
            {
                if (Console.KeyAvailable)
                    return Console.ReadKey(intercept: true).Key;
                await Task.Delay(50, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                string? line = await _input.ReadLineAsync(cancellationToken).ConfigureAwait(false);
                if (line is null)
                    return ConsoleKey.Q;
                if (!string.IsNullOrWhiteSpace(line))
                    return (ConsoleKey)char.ToUpperInvariant(line.Trim()[0]);
            }
        }
    }
}
