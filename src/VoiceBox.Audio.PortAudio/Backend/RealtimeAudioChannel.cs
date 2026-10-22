using System.Threading.Channels;

namespace VoiceBox.Audio.PortAudio.Backend;

internal static class RealtimeAudioChannel
{
    internal static BoundedChannelOptions CreateDropOldestOptions(
        int capacity,
        bool singleReader,
        bool singleWriter)
        // Dedicated audio consumers block on WaitToReadAsync; synchronous continuations let a
        // producer wake them without dispatching the signal through a potentially starved pool.
        => new(capacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = singleReader,
            SingleWriter = singleWriter,
            AllowSynchronousContinuations = true,
        };

    internal static Channel<T> CreateDropOldest<T>(
        int capacity,
        bool singleReader,
        bool singleWriter,
        Action<T>? itemDropped = null)
        => Channel.CreateBounded<T>(
            CreateDropOldestOptions(capacity, singleReader, singleWriter),
            itemDropped);

    internal static bool WaitToReadBlocking<T>(ChannelReader<T> reader, CancellationToken cancellationToken)
    {
        ValueTask<bool> wait = reader.WaitToReadAsync(cancellationToken);
        return wait.IsCompleted ? wait.Result : wait.AsTask().GetAwaiter().GetResult();
    }
}
