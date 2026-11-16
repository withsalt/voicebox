namespace VoiceBox.AEC.DTLN.Models
{
    /// <summary>DTLN-aec pretrained model size. Larger models cost more CPU but may suppress echo better.</summary>
    public enum DtlnAecModelSize
    {
        Units128 = 128,
        Units256 = 256,
        Units512 = 512,
    }

    internal static class DtlnAecModelSizeExtensions
    {
        public static int Units(this DtlnAecModelSize size)
            => size switch
            {
                DtlnAecModelSize.Units128 => 128,
                DtlnAecModelSize.Units256 => 256,
                DtlnAecModelSize.Units512 => 512,
                _ => throw new ArgumentOutOfRangeException(nameof(size), size, "Unsupported DTLN-aec model size."),
            };
    }
}
