using System.Diagnostics;

namespace AudioRecordDemo;

internal sealed class AudioLevelMeter
{
    private const int BarWidth = 30;
    private const double MinimumDb = -60.0;
    private static readonly long RefreshInterval = Stopwatch.Frequency / 10;
    private static readonly long BlinkInterval = Stopwatch.Frequency / 2;

    private readonly object _consoleLock = new();
    private long _nextRefreshTimestamp;
    private long _nextBlinkTimestamp;
    private bool _indicatorBright;
    private bool _hasOutput;
    private bool _cursorStateCaptured;
    private bool _originalCursorVisible;
    private bool _usesAnsiCursorControl;

    public void Update(ReadOnlySpan<float> samples)
    {
        if (samples.IsEmpty || Console.IsOutputRedirected)
        {
            return;
        }

        long now = Stopwatch.GetTimestamp();
        if (now < _nextRefreshTimestamp)
        {
            return;
        }

        _nextRefreshTimestamp = now + RefreshInterval;
        if (now >= _nextBlinkTimestamp)
        {
            _indicatorBright = !_indicatorBright;
            _nextBlinkTimestamp = now + BlinkInterval;
        }

        double sumOfSquares = 0;
        for (int i = 0; i < samples.Length; i++)
        {
            float sample = float.IsFinite(samples[i]) ? samples[i] : 0f;
            sumOfSquares += sample * sample;
        }

        double rms = Math.Sqrt(sumOfSquares / samples.Length);
        double db = Math.Clamp(20.0 * Math.Log10(Math.Max(rms, 1e-6)), MinimumDb, 0.0);
        int activeBars = (int)Math.Round((db - MinimumDb) / -MinimumDb * BarWidth);
        string bar = new string('#', activeBars) + new string('-', BarWidth - activeBars);

        lock (_consoleLock)
        {
            HideCursor();
            Console.Write('\r');
            ConsoleColor originalColor = Console.ForegroundColor;
            Console.ForegroundColor = _indicatorBright ? ConsoleColor.Red : ConsoleColor.DarkRed;
            Console.Write('●');
            Console.ForegroundColor = originalColor;
            Console.Write($" [{bar}] {db,6:F1} dBFS");
            _hasOutput = true;
        }
    }

    public void EndLine()
    {
        lock (_consoleLock)
        {
            if (_hasOutput)
            {
                Console.WriteLine();
                _hasOutput = false;
            }

            RestoreCursor();
        }
    }

    private void HideCursor()
    {
        if (_cursorStateCaptured)
        {
            return;
        }

        try
        {
            if (OperatingSystem.IsWindows())
            {
                _originalCursorVisible = Console.CursorVisible;
                Console.CursorVisible = false;
            }
            else
            {
                Console.Write("\u001b[?25l");
                _usesAnsiCursorControl = true;
            }

            _cursorStateCaptured = true;
        }
        catch (IOException)
        {
            // Some terminal hosts do not expose cursor visibility controls.
        }
    }

    private void RestoreCursor()
    {
        if (!_cursorStateCaptured)
        {
            return;
        }

        try
        {
            if (_usesAnsiCursorControl)
            {
                Console.Write("\u001b[?25h");
            }
            else if (OperatingSystem.IsWindows())
            {
                Console.CursorVisible = _originalCursorVisible;
            }
        }
        catch (IOException)
        {
            // The terminal may have been detached while recording.
        }

        _cursorStateCaptured = false;
        _usesAnsiCursorControl = false;
    }
}
