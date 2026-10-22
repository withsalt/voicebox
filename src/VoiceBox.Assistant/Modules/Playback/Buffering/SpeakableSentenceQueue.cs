using System.Threading.Channels;
using VoiceBox.Assistant.Modules.Response.Formatting;

namespace VoiceBox.Assistant.Modules.Playback.Buffering;

internal sealed class SpeakableSentenceQueue
{
    private readonly SentenceSplitter _splitter;
    private readonly Channel<SentenceItem> _sentences;

    public SpeakableSentenceQueue(int minSentenceChars)
    {
        _splitter = new SentenceSplitter(minSentenceChars);
        _sentences = Channel.CreateUnbounded<SentenceItem>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true,
            AllowSynchronousContinuations = false,
        });
    }

    public ChannelReader<SentenceItem> Reader => _sentences.Reader;

    public async ValueTask WriteAsync(string text, CancellationToken cancellationToken)
    {
        string speakable = TtsTextSanitizer.Clean(text);
        if (string.IsNullOrEmpty(speakable))
            return;

        foreach (string sentence in _splitter.Push(speakable))
            await _sentences.Writer.WriteAsync(
                new SentenceItem(sentence, Completion: null),
                cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask CompleteAsync(CancellationToken cancellationToken)
    {
        await FlushAsync(completion: null, cancellationToken).ConfigureAwait(false);
        _sentences.Writer.TryComplete();
    }

    public async ValueTask FlushAsync(
        TaskCompletionSource? completion,
        CancellationToken cancellationToken)
    {
        string? tail = _splitter.Flush();
        if (!string.IsNullOrWhiteSpace(tail))
        {
            await _sentences.Writer.WriteAsync(
                new SentenceItem(tail, completion),
                cancellationToken).ConfigureAwait(false);
            return;
        }

        if (completion is not null)
        {
            await _sentences.Writer.WriteAsync(
                new SentenceItem(Text: null, completion),
                cancellationToken).ConfigureAwait(false);
        }
    }

    public void Fail(Exception exception) => _sentences.Writer.TryComplete(exception);

    internal readonly record struct SentenceItem(
        string? Text,
        TaskCompletionSource? Completion);
}
