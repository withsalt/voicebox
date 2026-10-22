using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Gates;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class OperationCorrelationGateTests
{
    [Fact]
    public void ForgetTurn_CompactsExpiredCorrelationEntries()
    {
        var gate = new OperationCorrelationGate();

        for (long turnId = 1; turnId <= 4_100; turnId++)
        {
            Assert.True(gate.TryAdmit(
                turnId,
                operationId: turnId,
                AssistantSignalKind.RecognitionCompleted));
            gate.ForgetTurn(turnId);
        }

        Assert.Equal(4_096, gate.RetainedAdmissionCount);
        Assert.Equal(4_096, gate.RetainedOperationCount);
    }
}
