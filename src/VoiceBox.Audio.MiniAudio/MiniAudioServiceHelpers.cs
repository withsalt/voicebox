using System.Diagnostics;
using System.Numerics;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.MiniAudio;

internal static class MiniAudioServiceHelpers
{
    internal static void ScaleSamples(Span<float> samples, float gain)
    {
        int i = 0;
        int length = samples.Length;

        if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
        {
            var gains = new Vector<float>(gain);
            int vectorCount = Vector<float>.Count;
            int lastVectorStart = length - vectorCount;

            for (; i <= lastVectorStart; i += vectorCount)
            {
                var vector = new Vector<float>(samples.Slice(i, vectorCount));
                (vector * gains).CopyTo(samples.Slice(i, vectorCount));
            }
        }

        // Scalar tail loop
        for (; i < length; i++)
        {
            samples[i] *= gain;
        }
    }

    internal static float ApplyGainRamp(Span<float> samples, float current, float target, float step)
    {
        int length = samples.Length;
        float absoluteStep = MathF.Abs(step);

        // Keep ramp additions sequential to preserve per-sample rounding and state.
        for (int i = 0; i < length; i++)
        {
            if (MathF.Abs(target - current) > absoluteStep)
            {
                current += step;
            }
            else
            {
                // Once the ramp reaches its target, the remaining gain is constant.
                ScaleSamples(samples[i..], target);
                return target;
            }

            samples[i] *= current;
        }

        return current;
    }

    internal static void InvokeDataHandlers(
        Action<ReadOnlySpan<float>>? handlers,
        ReadOnlySpan<float> samples)
    {
        if (handlers is null)
        {
            return;
        }

        foreach (Action<ReadOnlySpan<float>> subscriber in Delegate.EnumerateInvocationList(handlers))
        {
            try
            {
                subscriber.Invoke(samples);
            }
            catch (Exception exception)
            {
                Debug.WriteLine($"MiniAudio capture subscriber failed: {exception.Message}");
            }
        }
    }

    internal static void InvokeDuplexFrameHandlers(
        DuplexAudioFrameHandler? handlers,
        in DuplexAudioFrame frame)
    {
        if (handlers is null)
        {
            return;
        }

        foreach (DuplexAudioFrameHandler subscriber in Delegate.EnumerateInvocationList(handlers))
        {
            try
            {
                subscriber.Invoke(in frame);
            }
            catch (Exception exception)
            {
                Debug.WriteLine($"MiniAudio duplex frame subscriber failed: {exception.Message}");
            }
        }
    }

    internal static void InvokeProcessedFrameHandlers(
        ProcessedCaptureFrameHandler? handlers,
        in ProcessedCaptureFrame frame)
    {
        if (handlers is null)
        {
            return;
        }

        foreach (ProcessedCaptureFrameHandler subscriber in Delegate.EnumerateInvocationList(handlers))
        {
            try
            {
                subscriber.Invoke(in frame);
            }
            catch (Exception exception)
            {
                Debug.WriteLine($"MiniAudio processed frame subscriber failed: {exception.Message}");
            }
        }
    }

    internal static void InvokeEventHandlers(EventHandler? handlers, object sender)
    {
        if (handlers is null)
        {
            return;
        }

        foreach (EventHandler subscriber in Delegate.EnumerateInvocationList(handlers))
        {
            try
            {
                subscriber.Invoke(sender, EventArgs.Empty);
            }
            catch (Exception exception)
            {
                Debug.WriteLine($"MiniAudio event subscriber failed: {exception.Message}");
            }
        }
    }

    internal static void InvokeEventHandlers<TEventArgs>(
        EventHandler<TEventArgs>? handlers,
        object sender,
        TEventArgs eventArgs)
        where TEventArgs : EventArgs
    {
        if (handlers is null)
        {
            return;
        }

        foreach (EventHandler<TEventArgs> subscriber in Delegate.EnumerateInvocationList(handlers))
        {
            try
            {
                subscriber.Invoke(sender, eventArgs);
            }
            catch (Exception exception)
            {
                Debug.WriteLine($"MiniAudio event subscriber failed: {exception.Message}");
            }
        }
    }
}
