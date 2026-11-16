using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.TurnContext;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class RecognitionAudioSnapshotTests
{
    [Fact]
    public void AudioTransfersOnceWithoutCrossingTurnBoundaries()
    {
        using var contexts = new TurnContextStore(new OperationCorrelationGate());
        var sink = (IRecognitionAudioSnapshot)contexts;
        contexts.TryCreate(1);
        contexts.TryCreate(2);
        sink.Store(1, new float[] { 0.1f, 0.2f });
        sink.Store(2, new float[] { -0.3f });

        Assert.Equal([0.1f, 0.2f], contexts.TakeRecognitionAudio(1).ToArray());
        Assert.True(contexts.TakeRecognitionAudio(1).IsEmpty);
        Assert.Equal([-0.3f], contexts.TakeRecognitionAudio(2).ToArray());
    }

    [Fact]
    public void LateSnapshotsCannotRecreateReleasedTurns()
    {
        using var contexts = new TurnContextStore(new OperationCorrelationGate());
        var sink = (IRecognitionAudioSnapshot)contexts;
        contexts.TryCreate(1);
        sink.Store(1, new float[] { 0.1f });
        contexts.Release(1);
        sink.Store(1, new float[] { 0.2f });

        Assert.True(contexts.TakeRecognitionAudio(1).IsEmpty);
        Assert.False(((ITurnResource)contexts).TryGet(1, out _));
    }
}
