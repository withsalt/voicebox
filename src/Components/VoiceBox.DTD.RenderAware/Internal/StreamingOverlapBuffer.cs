namespace VoiceBox.DTD.RenderAware.Internal;

/// <summary>Builds fixed two-hop windows in previous-then-current sample order.</summary>
internal sealed class StreamingOverlapBuffer
{
    private readonly int _hopLength;
    private readonly float[] _window;
    private bool _hasPreviousHop;

    public StreamingOverlapBuffer(int hopLength)
    {
        if (hopLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(hopLength));

        _hopLength = hopLength;
        _window = new float[checked(hopLength * 2)];
    }

    public ReadOnlySpan<float> Window => _window;
    internal float[] WindowBuffer => _window;

    public bool Append(ReadOnlySpan<float> hop)
    {
        if (hop.Length != _hopLength)
            throw new ArgumentException($"Hop length must be {_hopLength}.", nameof(hop));

        if (!_hasPreviousHop)
        {
            hop.CopyTo(_window.AsSpan(_hopLength));
            _hasPreviousHop = true;
            return false;
        }

        _window.AsSpan(_hopLength, _hopLength).CopyTo(_window);
        hop.CopyTo(_window.AsSpan(_hopLength));
        return true;
    }

    public void Reset()
    {
        Array.Clear(_window);
        _hasPreviousHop = false;
    }
}
