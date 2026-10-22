using Microsoft.Extensions.Options;
using VoiceBox.Audio.MiniAudio;
using VoiceBox.Audio.MiniAudio.Models;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace AudioRecordDemo;

internal static class Program
{
    private const int SampleRate = 16000;
    private const int ChannelCount = 1;

    private static async Task<int> Main(string[] args)
    {
        if (!TryParseArguments(args, out string outputPath, out int deviceIndex))
        {
            PrintUsage();
            return 1;
        }

        outputPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        BackendType backendType = BackendType.MiniAudio;
        IAudioRecordService? recorder = null;
        switch (backendType)
        {
            case BackendType.PortAudio:
                {
                    var playOptions = Options.Create(new PortAudioRecordOptions
                    {
                        SampleRate = SampleRate,
                        ChannelCount = ChannelCount,
                        SampleFormat = SampleFormat.Float32,
                    });
                    recorder = new PortAudioRecordService(playOptions);
                }
                break;
            case BackendType.MiniAudio:
                {
                    var playOptions = Options.Create(new MiniAudioRecordOptions
                    {
                        SampleRate = SampleRate,
                        ChannelCount = ChannelCount,
                        SampleFormat = SampleFormat.Float32,
                    });
                    recorder = new MiniAudioRecordService(playOptions);
                }
                break;
        }

        if (recorder == null) throw new ArgumentException("未知的后端类型");
        Console.WriteLine($"使用音频后端：{backendType}");

        using var writer = new Pcm16WaveWriter(outputPath, SampleRate, ChannelCount);
        var levelMeter = new AudioLevelMeter();
        var stopped = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        ConsoleCancelEventHandler cancelHandler = (_, e) =>
        {
            e.Cancel = true;
            stopped.TrySetResult();
        };
        Console.CancelKeyPress += cancelHandler;

        recorder.OnDataAvailable += writer.WriteSamples;
        recorder.OnDataAvailable += levelMeter.Update;
        recorder.OnError += (_, e) =>
        {
            levelMeter.EndLine();
            Console.Error.WriteLine($"录音错误: {e.Exception.Message}");
            stopped.TrySetException(e.Exception);
        };
        recorder.OnXrun += (_, e) =>
        {
            levelMeter.EndLine();
            Console.Error.WriteLine($"录音缓冲溢出: {e}");
        };

        try
        {
            PrintInputDevices(recorder);
            await recorder.StartRecordingAsync(deviceIndex);

            Console.WriteLine($"正在录音: {SampleRate} Hz, {ChannelCount} 声道, PCM 16-bit");
            Console.WriteLine($"输出文件: {outputPath}");
            Console.WriteLine("按 Enter 或 Ctrl+C 停止录音。");

            Task enterPressed = Task.Run(Console.ReadLine);
            Task completed = await Task.WhenAny(enterPressed, stopped.Task);
            if (completed == stopped.Task)
            {
                await stopped.Task;
            }

            await recorder.StopRecordingAsync();
            levelMeter.EndLine();
            writer.FinalizeHeader();

            Console.WriteLine($"录音已保存，共 {writer.Duration.TotalSeconds:F2} 秒: {outputPath}");
            return 0;
        }
        catch (Exception ex)
        {
            levelMeter.EndLine();
            Console.Error.WriteLine($"录音失败: {ex.Message}");
            return 1;
        }
        finally
        {
            Console.CancelKeyPress -= cancelHandler;

            if (recorder.IsRecording)
            {
                try
                {
                    await recorder.StopRecordingAsync();
                }
                catch
                {
                    // The original error is reported above.
                }
            }
        }
    }

    private static bool TryParseArguments(string[] args, out string outputPath, out int deviceIndex)
    {
        outputPath = args.Length > 0
            ? args[0]
            : Path.Combine(Environment.CurrentDirectory, $"recording-{DateTime.Now:yyyyMMdd-HHmmss}.wav");
        deviceIndex = -1;

        if (args.Length > 2 || !outputPath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return args.Length < 2 || int.TryParse(args[1], out deviceIndex);
    }

    private static void PrintInputDevices(IAudioRecordService recorder)
    {
        Console.WriteLine("可用输入设备:");
        foreach (AudioDevice device in recorder.ListAudioDevices(AudioDeviceType.Input) ?? [])
        {
            Console.WriteLine($"  {device.Id}. {device.Name}{(device.IsDefault ? " [默认]" : string.Empty)}");
        }

        Console.WriteLine();
    }

    private static void PrintUsage()
    {
        Console.Error.WriteLine("用法: dotnet run --project src/Demos/AudioRecordDemo -- [输出.wav] [输入设备编号]");
    }
}
