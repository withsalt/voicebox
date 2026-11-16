namespace VoiceBox.ASR.FireRedASR2.Internal.Model
{
    /// <summary>
    /// CTC head output: flat layout <c>[batch * frameCount * vocabSize]</c>. The legacy
    /// representation was <c>List&lt;List&lt;float[]&gt;&gt;</c> which produced one tiny
    /// array per frame and crawled through tensor indexers.
    /// </summary>
    internal sealed class CtcOutputEntity
    {
        public float[]? Logits { get; set; }

        public int Batch { get; set; }

        public int FrameCount { get; set; }

        public int VocabSize { get; set; }

        public bool HasData => Logits is not null && FrameCount > 0 && VocabSize > 0;
    }
}
