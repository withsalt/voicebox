using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class SpeakerProfileMatcherTests
{
    private readonly ScriptedSpeakerService _speaker = new();

    [Fact]
    public void Score_ReportsCentroidBestAndTop2()
    {
        SpeakerPrototypeBank bank = SpeakerTestProfiles.OwnerBank();
        float[] embedding = ScriptedSpeakerService.EmbeddingWithOwnerScore(0.8f);
        SpeakerWindowScore score = SpeakerProfileMatcher.Score(_speaker, bank, embedding);

        float p1 = _speaker.CompareEmbeddings(embedding, bank.Prototypes[0].Embedding);
        float p2 = _speaker.CompareEmbeddings(embedding, bank.Prototypes[1].Embedding);
        Assert.Equal(0.8f, score.CentroidScore, 3);
        Assert.Equal(Math.Max(p1, p2), score.BestPrototypeScore, 3);
        Assert.Equal((p1 + p2) / 2f, score.Top2MeanScore, 3);
        Assert.Equal(Math.Max(score.Top2MeanScore, score.CentroidScore), score.Reported, 3);
    }

    [Fact]
    public void SingleOutlyingPrototype_CannotAuthorizeAlone()
    {
        // Two prototypes: one owner axis, one pointing at the impostor. An impostor embedding hits the
        // outlier prototype with 1.0 but top-2 mean is 0.5 and the centroid disagrees.
        var bank = new SpeakerPrototypeBank(
            ScriptedSpeakerService.OwnerAxis(),
            [
                new SpeakerPrototype("owner", ScriptedSpeakerService.OwnerAxis(), 1f),
                new SpeakerPrototype("outlier", ScriptedSpeakerService.OtherAxis(), 0.2f),
            ]);
        SpeakerWindowScore score = SpeakerProfileMatcher.Score(_speaker, bank, ScriptedSpeakerService.OtherAxis());

        Assert.Equal(1f, score.BestPrototypeScore, 3);
        Assert.Equal(0.5f, score.Top2MeanScore, 3);
        Assert.Equal(0f, score.CentroidScore, 3);
        Assert.False(SpeakerProfileMatcher.IsMatch(in score, threshold: 0.6f));
        Assert.False(SpeakerProfileMatcher.IsHighConfidence(in score, 0.6f, 0.75f));
    }

    [Fact]
    public void ThresholdComparison_UsesGreaterOrEqual()
    {
        SpeakerPrototypeBank bank = SpeakerTestProfiles.OwnerBank();
        var exactly = new SpeakerWindowScore(CentroidScore: 0.5f, BestPrototypeScore: 0.5f, Top2MeanScore: 0.5f);
        Assert.True(SpeakerProfileMatcher.IsMatch(in exactly, 0.5f));
        var below = new SpeakerWindowScore(0.4999f, 0.4999f, 0.4999f);
        Assert.False(SpeakerProfileMatcher.IsMatch(in below, 0.5f));
        Assert.True(SpeakerProfileMatcher.IsHighConfidence(new SpeakerWindowScore(0.5f, 0.9f, 0.65f), 0.5f, 0.65f));
        Assert.False(SpeakerProfileMatcher.IsHighConfidence(new SpeakerWindowScore(0.4f, 0.9f, 0.9f), 0.5f, 0.65f));
        Assert.False(SpeakerProfileMatcher.IsHighConfidence(new SpeakerWindowScore(0.9f, 0.9f, 0.9f), 0.5f, 0.5f));
        _ = bank;
    }

    [Fact]
    public void Rejected_RequiresClearingTheIndeterminateMargin()
    {
        var ambiguous = new SpeakerWindowScore(0.46f, 0.46f, 0.46f);
        Assert.False(SpeakerProfileMatcher.IsMatch(in ambiguous, 0.5f));
        Assert.False(SpeakerProfileMatcher.IsRejected(in ambiguous, 0.5f, 0.05f));
        var clearlyBelow = new SpeakerWindowScore(0.2f, 0.2f, 0.2f);
        Assert.True(SpeakerProfileMatcher.IsRejected(in clearlyBelow, 0.5f, 0.05f));
    }

    [Fact]
    public void NonFiniteScores_NeverMatch()
    {
        var nan = new SpeakerWindowScore(float.NaN, 1f, 1f);
        Assert.False(SpeakerProfileMatcher.IsMatch(in nan, 0.5f));
        Assert.False(SpeakerProfileMatcher.IsRejected(in nan, 0.5f, 0.05f));
    }

    [Fact]
    public void Bank_WithSinglePrototypeIsUnusable()
    {
        var bank = new SpeakerPrototypeBank(
            ScriptedSpeakerService.OwnerAxis(),
            [new SpeakerPrototype("only", ScriptedSpeakerService.OwnerAxis(), 1f)]);
        Assert.False(bank.IsUsable);
        Assert.Throws<InvalidOperationException>(() =>
            SpeakerProfileMatcher.Score(_speaker, bank, ScriptedSpeakerService.OwnerAxis()));
    }
}
