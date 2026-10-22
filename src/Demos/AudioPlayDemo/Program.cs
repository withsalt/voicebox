using Microsoft.Extensions.Options;
using VoiceBox.Audio.Common;
using VoiceBox.Audio.Common.AudioReader;
using VoiceBox.Audio.MiniAudio;
using VoiceBox.Audio.MiniAudio.Models;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace AudioPlayDemo;

internal static class Program
{
    private const int SampleRate = 24000;
    private const int ChannelCount = 1;
    private const int BufferSize = 4096;
    private const int MaxBufferedChunks = 8;

    private static async Task Main()
    {
        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cancellation.Cancel();
        };

        string samplesDirectory = Path.Combine(AppContext.BaseDirectory, "samples");
        var tracks = new[]
        {
            new Track("sample.wav", () => OpenWave(Path.Combine(samplesDirectory, "sample.wav"))),
            new Track("sample.pcm", () => OpenPcm(Path.Combine(samplesDirectory, "sample.pcm"))),
            new Track("sample.opus", () => OpenOpus(Path.Combine(samplesDirectory, "sample.opus"))),
        };

        BackendType backendType = BackendType.MiniAudio;
        IAudioPlayService? player = null;
        switch (backendType)
        {
            case BackendType.PortAudio:
                {
                    var playOptions = Options.Create(new PortAudioPlayOptions
                    {
                        SampleRate = SampleRate,
                        ChannelCount = ChannelCount,
                        SampleFormat = SampleFormat.Float32,
                        MaxQueueSize = 16,
                    });
                    player = new PortAudioPlayService(playOptions);
                }
                break;
            case BackendType.MiniAudio:
                {
                    var playOptions = Options.Create(new MiniAudioPlayOptions
                    {
                        SampleRate = SampleRate,
                        ChannelCount = ChannelCount,
                        SampleFormat = SampleFormat.Float32,
                        MaxQueueSize = 16,
                    });
                    player = new MiniAudioPlayService(playOptions);
                }
                break;
        }

        if (player == null) throw new ArgumentException("未知的后端类型");
        Console.WriteLine($"使用音频后端：{backendType}");

        try
        {
            PrintOutputDevices(player);

            await player.StartPlayingAsync(cancellationToken: cancellation.Token);

            foreach (var track in tracks)
            {
                Console.WriteLine($"正在播放 {track.Name}...");
                using var reader = track.OpenReader();
                long sampleCount = await PlayStreamAsync(player, reader, cancellation.Token);
                Console.WriteLine($"{track.Name} 播放完成（{sampleCount / (double)SampleRate:F2} 秒）。");
            }

            Console.WriteLine("三个音频文件已按顺序播放完毕。");
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("播放已取消。");
        }
        finally
        {
            await player.StopPlayingAsync();
            player.Dispose();
        }
    }

    private static IAudioStreamReader OpenWave(string path)
    {
        EnsureFileExists(path);
        var reader = new WaveStreamReader(path);

        if (reader.FormatChunk.SampleRate != SampleRate || reader.FormatChunk.NumChannels != ChannelCount)
        {
            reader.Dispose();
            throw new InvalidDataException(
                $"{Path.GetFileName(path)} 必须是 {SampleRate} Hz、{ChannelCount} 声道音频。");
        }

        return reader;
    }

    private static IAudioStreamReader OpenPcm(string path)
    {
        EnsureFileExists(path);
        return new PcmStreamReader(path, bitsPerSample: 16, sampleFormat: SampleFormat.Int16);
    }

    private static IAudioStreamReader OpenOpus(string path)
    {
        EnsureFileExists(path);
        return new OpusStreamReader(path, SampleRate, ChannelCount);
    }

    private static async Task<long> PlayStreamAsync(
        IAudioPlayService player,
        IAudioStreamReader reader,
        CancellationToken cancellationToken)
    {
        var completed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int inputCompleted = 0;
        long sampleCount = 0;

        EventHandler completionHandler = (_, _) =>
        {
            if (Volatile.Read(ref inputCompleted) != 0)
            {
                completed.TrySetResult();
            }
        };
        EventHandler<AudioErrorEventArgs> errorHandler = (_, e) => completed.TrySetException(e.Exception);

        player.OnPlaybackCompleted += completionHandler;
        player.OnError += errorHandler;

        try
        {
            var buffer = new float[BufferSize];
            int samplesRead;

            while ((samplesRead = reader.ReadSamples(buffer)) > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();

                while (player.CurrentQueueSize >= MaxBufferedChunks)
                {
                    if (completed.Task.IsFaulted)
                    {
                        await completed.Task;
                    }

                    await Task.Delay(5, cancellationToken);
                }

                while (!player.TryEnqueue(buffer.AsSpan(0, samplesRead), out _))
                {
                    if (completed.Task.IsFaulted)
                    {
                        await completed.Task;
                    }

                    await Task.Delay(5, cancellationToken);
                }

                sampleCount += samplesRead;
            }

            Volatile.Write(ref inputCompleted, 1);

            // 若最后一块恰好在设置完成标记前播放完，完成事件可能已被忽略；
            // 此处补查播放器状态，避免等待一个不会再次触发的事件。
            if (!player.IsPlaying && player.CurrentQueueSize == 0)
            {
                completed.TrySetResult();
            }

            await completed.Task.WaitAsync(cancellationToken);
            return sampleCount;
        }
        finally
        {
            player.OnPlaybackCompleted -= completionHandler;
            player.OnError -= errorHandler;
        }
    }

    private static void PrintOutputDevices(IAudioPlayService recorder)
    {
        Console.WriteLine("可用播放设备:");
        foreach (AudioDevice device in recorder.ListAudioDevices(AudioDeviceType.Output) ?? [])
        {
            Console.WriteLine($"  {device.Id}. {device.Name}{(device.IsDefault ? " [默认]" : string.Empty)}");
        }

        Console.WriteLine();
    }

    private static void EnsureFileExists(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("找不到示例音频文件。", path);
        }
    }

    private sealed record Track(string Name, Func<IAudioStreamReader> OpenReader);
}
