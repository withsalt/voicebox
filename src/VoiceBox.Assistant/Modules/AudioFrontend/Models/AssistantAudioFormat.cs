using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.Playback.Contracts;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Models
{
    /// <summary>
    /// Resolves the runtime capture/playback format and performs duration/sample conversions in one place.
    /// </summary>
    internal sealed class AssistantAudioFormat
    {
        private readonly IAssistantAudioInput _input;
        private readonly ISpeechSynthesisStreamPort _speechSynthesis;

        public AssistantAudioFormat(IAssistantAudioInput input, ISpeechSynthesisStreamPort speechSynthesis)
        {
            _input = input ?? throw new ArgumentNullException(nameof(input));
            _speechSynthesis = speechSynthesis ?? throw new ArgumentNullException(nameof(speechSynthesis));
        }

        public int AudioSampleRate
            => _input.SampleRate is > 0 and <= int.MaxValue
                ? (int)_input.SampleRate
                : _speechSynthesis.OutputSampleRate;

        public int AudioChannels
            => _input.ChannelCount is > 0 and <= int.MaxValue
                ? (int)_input.ChannelCount
                : Math.Max(1, _speechSynthesis.OutputChannels);

        public int WakeupSampleRate => AudioSampleRate;

        public int WakeupChannels => AudioChannels;

        public int SamplesFromMillisecondsToInt32(int milliseconds, int sampleRate, int channels)
        {
            long samples = SamplesFromMilliseconds(milliseconds, sampleRate, channels);
            return samples > int.MaxValue ? int.MaxValue : (int)samples;
        }

        public double SamplesToMilliseconds(long samples)
        {
            int sampleRate = AudioSampleRate;
            int channels = AudioChannels;
            if (samples <= 0 || sampleRate <= 0 || channels <= 0)
                return 0;

            return samples * 1000d / sampleRate / channels;
        }

        public static long SamplesFromMilliseconds(int milliseconds, int sampleRate, int channels)
        {
            if (milliseconds <= 0 || sampleRate <= 0 || channels <= 0)
                return 0;

            return (long)milliseconds * sampleRate * channels / 1000;
        }
    }
}
