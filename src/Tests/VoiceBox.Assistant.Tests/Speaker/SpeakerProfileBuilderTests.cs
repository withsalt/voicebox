using VoiceBox.Assistant.Modules.Speaker.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Enrollment;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Assistant.Modules.Speaker.Profiles;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class SpeakerProfileBuilderTests
{
    private readonly ScriptedSpeakerService _speaker = new();

    [Theory]
    [InlineData(SpeakerEnrollmentCondition.Clean, 39)]
    [InlineData(SpeakerEnrollmentCondition.Clean, 40)]
    [InlineData(SpeakerEnrollmentCondition.DoubleTalk, 39)]
    [InlineData(SpeakerEnrollmentCondition.DoubleTalk, 40)]
    public void Coverage_PrefersUnrepresentedDirectionOverDuplicateCluster(
        SpeakerEnrollmentCondition condition,
        double duplicateAngle)
    {
        // Directions stand in for recording conditions within one bank. The medoid is 0 degrees;
        // after covering -40 and +40, the fourth slot should cover +/-20, not the cluster at +40.
        SpeakerEnrollmentSegmentEmbedding[] segments = Segments(condition, 0, -40, 40, duplicateAngle, 20, -20);
        var builder = new SpeakerProfileBuilder(_speaker, new SpeakerEnrollmentOptions());

        foreach (SpeakerEnrollmentSegmentEmbedding[] ordered in new[] { segments, segments.Reverse().ToArray() })
        {
            SpeakerBankBuildResult result = Assert.IsType<SpeakerBankBuildResult>(builder.TryBuildBank(ordered, minRetained: 4));

            Assert.Equal(6, result.Retained.Count);
            Assert.Empty(result.Discarded);
            Assert.Equal(4, result.Bank.Prototypes.Count);
            Assert.Contains(result.Bank.Prototypes, p => ReferenceEquals(p.Embedding, segments[0].Embedding));
            Assert.Contains(result.Bank.Prototypes, p => ReferenceEquals(p.Embedding, segments[1].Embedding));
            // Either copy can represent +40 when the duplicate is exact.
            Assert.Single(result.Bank.Prototypes, p =>
                ReferenceEquals(p.Embedding, segments[2].Embedding) || ReferenceEquals(p.Embedding, segments[3].Embedding));
            Assert.Single(result.Bank.Prototypes, p =>
                ReferenceEquals(p.Embedding, segments[4].Embedding) || ReferenceEquals(p.Embedding, segments[5].Embedding));
        }
    }

    [Theory]
    [InlineData(SpeakerEnrollmentCondition.Clean, 0)]
    [InlineData(SpeakerEnrollmentCondition.Clean, 0.05)]
    [InlineData(SpeakerEnrollmentCondition.DoubleTalk, 0)]
    [InlineData(SpeakerEnrollmentCondition.DoubleTalk, 0.05)]
    public void DuplicateDirections_DoNotFillUnusedSlotsOrDiscardAcceptedSegments(
        SpeakerEnrollmentCondition condition,
        double perturbation)
    {
        SpeakerEnrollmentSegmentEmbedding[] segments = Segments(condition, 0, perturbation, 20, 20 + perturbation, -20, -20 + perturbation);
        var builder = new SpeakerProfileBuilder(_speaker, new SpeakerEnrollmentOptions { MaxPrototypesPerBank = 8 });

        SpeakerBankBuildResult result = Assert.IsType<SpeakerBankBuildResult>(builder.TryBuildBank(segments, minRetained: 4));

        Assert.True(result.Bank.IsUsable);
        Assert.Equal(3, result.Bank.Prototypes.Count);
        for (int start = 0; start < segments.Length; start += 2)
        {
            Assert.Single(result.Bank.Prototypes, p =>
                ReferenceEquals(p.Embedding, segments[start].Embedding) || ReferenceEquals(p.Embedding, segments[start + 1].Embedding));
        }
        Assert.Equal(segments, result.Retained);
        Assert.Empty(result.Discarded);
        // Suppression only affects representative selection: all accepted recordings still support
        // the centroid and enrollment statistics.
        float[] expectedCentroid = SpeakerEmbeddingMath.NormalizedMean(segments.Select(s => s.Embedding).ToArray());
        Assert.Equal(expectedCentroid, result.Bank.Centroid);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void IdenticalRecordings_PreserveMinimumDistinctSegmentsForTop2Matching(int minimum)
    {
        SpeakerEnrollmentSegmentEmbedding[] segments = Segments(SpeakerEnrollmentCondition.Clean, 0, 0, 0, 0, 0);
        var builder = new SpeakerProfileBuilder(_speaker, new SpeakerEnrollmentOptions { MinPrototypesPerBank = minimum });

        SpeakerBankBuildResult result = Assert.IsType<SpeakerBankBuildResult>(builder.TryBuildBank(segments, minRetained: 4));

        Assert.True(result.Bank.IsUsable);
        Assert.Equal(minimum, result.Bank.Prototypes.Count);
        Assert.Equal(minimum, result.Bank.Prototypes.Select(p => p.Id).Distinct().Count());
        Assert.Equal(5, result.Retained.Count);
        Assert.Empty(result.Discarded);
        SpeakerWindowScore score = SpeakerProfileMatcher.Score(_speaker, result.Bank, segments[0].Embedding);
        Assert.True(SpeakerProfileMatcher.IsMatch(in score, _speaker.Threshold));
    }

    [Fact]
    public void SimilarButDistinctDirections_AreNotMergedWhenSlotsAreAvailable()
    {
        SpeakerEnrollmentSegmentEmbedding[] segments = Segments(SpeakerEnrollmentCondition.Clean, 0, 0.2, 20, -20);
        var builder = new SpeakerProfileBuilder(_speaker, new SpeakerEnrollmentOptions { MaxPrototypesPerBank = 8 });

        SpeakerBankBuildResult result = Assert.IsType<SpeakerBankBuildResult>(builder.TryBuildBank(segments, minRetained: 4));

        Assert.Equal(4, result.Bank.Prototypes.Count);
        foreach (SpeakerEnrollmentSegmentEmbedding segment in segments)
            Assert.Contains(result.Bank.Prototypes, p => ReferenceEquals(p.Embedding, segment.Embedding));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void Coverage_RespectsPrototypeBudget(int maximum)
    {
        SpeakerEnrollmentSegmentEmbedding[] segments = Segments(SpeakerEnrollmentCondition.Clean, 0, -20, 20, -10, 10);
        var builder = new SpeakerProfileBuilder(_speaker, new SpeakerEnrollmentOptions { MaxPrototypesPerBank = maximum });

        SpeakerBankBuildResult result = Assert.IsType<SpeakerBankBuildResult>(builder.TryBuildBank(segments, minRetained: 4));

        Assert.Equal(maximum, result.Bank.Prototypes.Count);
        Assert.Equal(maximum, result.Bank.Prototypes.Select(p => p.Id).Distinct().Count());
    }

    [Fact]
    public void Coverage_DoesNotReintroduceAnOutlierForDiversity()
    {
        SpeakerEnrollmentSegmentEmbedding[] segments = Segments(SpeakerEnrollmentCondition.Clean, 0, -10, 10, 10.02, 5, -5, 120);
        var builder = new SpeakerProfileBuilder(_speaker, new SpeakerEnrollmentOptions());

        SpeakerBankBuildResult result = Assert.IsType<SpeakerBankBuildResult>(builder.TryBuildBank(segments, minRetained: 4));

        Assert.Same(segments[6], Assert.Single(result.Discarded));
        Assert.Equal(6, result.Retained.Count);
        Assert.Equal(4, result.Bank.Prototypes.Count);
        Assert.DoesNotContain(result.Bank.Prototypes, p => ReferenceEquals(p.Embedding, segments[6].Embedding));
    }

    private static SpeakerEnrollmentSegmentEmbedding[] Segments(SpeakerEnrollmentCondition condition, params double[] angles)
        => angles.Select((angle, index) =>
        {
            double radians = angle * Math.PI / 180;
            var embedding = new float[8];
            embedding[0] = (float)Math.Cos(radians);
            embedding[1] = (float)Math.Sin(radians);
            return new SpeakerEnrollmentSegmentEmbedding(
                condition, index, embedding, [embedding], SpeakerEnrollmentSegmentQuality.Empty);
        }).ToArray();
}
