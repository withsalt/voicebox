using Misaki.NET.Jieba.Segmenter;
using Misaki.NET.Jieba.Segmenter.PosSeg;
using Xunit;

namespace Misaki.Tests;

/// <summary>
/// Tests covering #5: the removal of `buf += w` in JiebaSegmenter.CutDag /
/// CutDagWithoutHmm and PosSegmenter.CutDag / CutDagWithoutHmm.
///
/// Behaviour is asserted on the public Cut APIs — the buffer-accumulation
/// logic now uses (bufStart, bufLen) indices, and the emitted tokens must
/// match the prior behaviour exactly.
/// </summary>
public class BufferAccumulationTests
{
    [Fact]
    public void JiebaCut_HmmEnabled_SingleCharBufferFlushedCorrectly()
    {
        var seg = new JiebaSegmenter();
        // A chain of single-char DAG steps plus a multi-char word forces the
        // bufLen path to be flushed by AddBufferToWordList.
        var tokens = seg.Cut("他来到了北京", cutAll: false, hmm: true).ToList();
        Assert.NotEmpty(tokens);
        Assert.Equal("他来到了北京", string.Concat(tokens));
        Assert.Contains("北京", tokens);
    }

    [Fact]
    public void JiebaCut_HmmDisabled_SingleCharBufferFlushedCorrectly()
    {
        var seg = new JiebaSegmenter();
        var tokens = seg.Cut("他来到了北京", cutAll: false, hmm: false).ToList();
        Assert.Equal("他来到了北京", string.Concat(tokens));
    }

    [Fact]
    public void JiebaCut_AsciiRunInsideCjk_StaysGrouped()
    {
        var seg = new JiebaSegmenter();
        // CutDagWithoutHmm groups consecutive English chars into one token.
        var tokens = seg.Cut("使用iPhone16很好", cutAll: false, hmm: false).ToList();
        Assert.Equal("使用iPhone16很好", string.Concat(tokens));
        // The ASCII run should appear as a single "iPhone16" (or at minimum iPhone16 as one token)
        Assert.Contains(tokens, t => t.Contains("iPhone16") || t == "iPhone16");
    }

    [Fact]
    public void JiebaCut_EndsOnBuffer_FlushesTrailingBuffer()
    {
        var seg = new JiebaSegmenter();
        // Sentence ending in a run of single-char DAG entries so buffer flush happens at the tail.
        var tokens = seg.Cut("好吗", cutAll: false, hmm: true).ToList();
        Assert.Equal("好吗", string.Concat(tokens));
    }

    [Fact]
    public void JiebaCut_OnlySingleCharFragments_AllCoveredByBuffer()
    {
        var seg = new JiebaSegmenter();
        var tokens = seg.Cut("啊啊啊", cutAll: false, hmm: false).ToList();
        // No matter how they're grouped, concatenation must match.
        Assert.Equal("啊啊啊", string.Concat(tokens));
    }

    [Fact]
    public void PosSegCut_HmmEnabled_TokensRoundTrip()
    {
        var pos = new PosSegmenter();
        var pairs = pos.Cut("他来到了北京", hmm: true).ToList();
        Assert.Equal("他来到了北京", string.Concat(pairs.Select(p => p.Word)));
    }

    [Fact]
    public void PosSegCut_HmmDisabled_TokensRoundTrip()
    {
        var pos = new PosSegmenter();
        var pairs = pos.Cut("他来到了北京", hmm: false).ToList();
        Assert.Equal("他来到了北京", string.Concat(pairs.Select(p => p.Word)));
    }

    [Fact]
    public void PosSegCut_AsciiRunTaggedEng()
    {
        var pos = new PosSegmenter();
        var pairs = pos.Cut("学习Python很有趣", hmm: false).ToList();
        // Ensure the ASCII block survives as one token tagged "eng".
        Assert.Contains(pairs, p => p.Word == "Python" && p.Flag == "eng");
        Assert.Equal("学习Python很有趣", string.Concat(pairs.Select(p => p.Word)));
    }

    [Fact]
    public void PosSegCut_TrailingAsciiRun_FlushedAsEng()
    {
        var pos = new PosSegmenter();
        var pairs = pos.Cut("看Python", hmm: false).ToList();
        Assert.Contains(pairs, p => p.Word == "Python" && p.Flag == "eng");
        Assert.Equal("看Python", string.Concat(pairs.Select(p => p.Word)));
    }

    [Fact]
    public void PosSegCut_MixedBufferFlush_NoDuplicationOrLoss()
    {
        var pos = new PosSegmenter();
        const string s = "ABC你好DEF世界GHI";
        var pairs = pos.Cut(s, hmm: false).ToList();
        Assert.Equal(s, string.Concat(pairs.Select(p => p.Word)));
    }
}
