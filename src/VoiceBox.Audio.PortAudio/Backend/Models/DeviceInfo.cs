// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>

using System.Runtime.InteropServices;
using System.Text;

namespace VoiceBox.Audio.PortAudio.Backend.Models
{
    [StructLayout(LayoutKind.Sequential)]
    public struct DeviceInfoData
    {
        public int structVersion;

        public nint name;

        public int hostApi;        // note this is a host API index, not a type id

        public int maxInputChannels;
        public int maxOutputChannels;

        // Default latency values for interactive use.
        public double defaultLowInputLatency;
        public double defaultLowOutputLatency;

        // Default latency values for non-interactive use (for example, playing sound files).
        public double defaultHighInputLatency;
        public double defaultHighOutputLatency;

        public double defaultSampleRate;

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("DeviceInfo [");
            sb.AppendLine($"  structVersion={structVersion}");
            sb.AppendLine($"  name={name}");
            sb.AppendLine($"  hostApi={hostApi}");
            sb.AppendLine($"  maxInputChannels={maxInputChannels}");
            sb.AppendLine($"  maxOutputChannels={maxOutputChannels}");
            sb.AppendLine($"  defaultSampleRate={defaultSampleRate}");
            sb.AppendLine($"  defaultLowInputLatency={defaultLowInputLatency}");
            sb.AppendLine($"  defaultLowOutputLatency={defaultLowOutputLatency}");
            sb.AppendLine($"  defaultHighInputLatency={defaultHighInputLatency}");
            sb.AppendLine($"  defaultHighOutputLatency={defaultHighOutputLatency}");
            sb.AppendLine($"  defaultHighSampleRate={defaultSampleRate}");
            sb.AppendLine("]");
            return sb.ToString();
        }
    }

    /// <summary>
    /// A structure providing information and capabilities of PortAudio devices.
    /// Devices may support input, output or both input and output.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DeviceInfo
    {
        public int structVersion;           // this is struct version 2

        [MarshalAs(UnmanagedType.LPStr)]
        public string name;                 // Originally: `const char *`

        public int hostApi;        // note this is a host API index, not a type id

        public int maxInputChannels;
        public int maxOutputChannels;

        // Default latency values for interactive use.
        public double defaultLowInputLatency;
        public double defaultLowOutputLatency;

        // Default latency values for non-interactive use (for example, playing sound files).
        public double defaultHighInputLatency;
        public double defaultHighOutputLatency;

        public double defaultSampleRate;

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("DeviceInfo [");
            sb.AppendLine($"  structVersion={structVersion}");
            sb.AppendLine($"  name={name}");
            sb.AppendLine($"  hostApi={hostApi}");
            sb.AppendLine($"  maxInputChannels={maxInputChannels}");
            sb.AppendLine($"  maxOutputChannels={maxOutputChannels}");
            sb.AppendLine($"  defaultSampleRate={defaultSampleRate}");
            sb.AppendLine($"  defaultLowInputLatency={defaultLowInputLatency}");
            sb.AppendLine($"  defaultLowOutputLatency={defaultLowOutputLatency}");
            sb.AppendLine($"  defaultHighInputLatency={defaultHighInputLatency}");
            sb.AppendLine($"  defaultHighOutputLatency={defaultHighOutputLatency}");
            sb.AppendLine($"  defaultHighSampleRate={defaultSampleRate}");
            sb.AppendLine("]");
            return sb.ToString();
        }
    }
}
