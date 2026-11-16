using VoiceBox.ASR.FireRedASR2.Internal.Model;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Strategy for converting decoded token-ids (plus per-token timestamps) into the
    /// final recognised text. Different model families use different sub-word schemes
    /// (BPE, WordPiece, raw Chinese characters); each implementation handles its scheme.
    /// </summary>
    internal interface ITokenAssembler
    {
        OfflineRecognizerResultEntity Assemble(IReadOnlyList<long> tokens, IReadOnlyList<int[]> timestamps);
    }
}
