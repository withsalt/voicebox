using System.Runtime.InteropServices;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Backend.Models
{
    public struct HostApiInfo
    {
        public int structVersion;

        public HostApiType type;

        [MarshalAs(UnmanagedType.LPStr)]
        public string name;

        public int deviceCount;

        public int defaultInputDevice;

        public int defaultOutputDevice;

        public override string ToString() =>
            $"[{GetType().Name}]\n" +
            $"  structVersion: {structVersion}\n" +
            $"  type: {type} ({(int)type})\n" +
            $"  name: {name}\n" +
            $"  deviceCount: {deviceCount}\n" +
            $"  defaultInputDevice: {defaultInputDevice}\n" +
            $"  defaultOutputDevice: {defaultOutputDevice}";
    }
}
