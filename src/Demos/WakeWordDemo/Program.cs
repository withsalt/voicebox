using Microsoft.Extensions.Logging;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Wakeup.OpenWakeWord;

namespace WakeWordDemo
{
    /// <summary>
    /// 演示 OpenWakeWord 流式唤醒词检测。
    /// 默认加载随包提供的 xiao_ai (小爱) 中文唤醒词模型, 采样率固定 16kHz / 单声道。
    /// 程序启动后开始持续监听麦克风, 命中唤醒词时输出事件。按 Q / ESC 退出。
    /// </summary>
    internal class Program
    {
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";

        private static readonly string ModelDir = Path.Combine(BASE_MODEL_DIR, "kws", "openWakeWord", "preset", "alexa_v0.1.onnx");

        static async Task Main()
        {
            using var audioManager = new AudioServiceManager()
                .AddLogger(b =>
                {
                    b.SetMinimumLevel(LogLevel.Information);
                    b.AddSimpleConsole(o =>
                    {
                        o.SingleLine = true;
                        o.TimestampFormat = "HH:mm:ss ";
                    });
                })
                .AddPortAudioRecordService(options =>
                {
                    options.SampleRate = 16000;
                    options.ChannelCount = 1;
                    options.SampleFormat = SampleFormat.Float32;
                })
                .AddOpenWakeWordService(options =>
                {
                    options.SampleRate = 16000;
                    options.ChannelCount = 1;
                    options.AddWakeWord(
                        keyword: "Alexa",
                        modelPath: ModelDir,
                        threshold: 0.5f,
                        debounceMs: 1500);
                })
                .Build();

            using var recordService = audioManager.GetAudioRecordInstance();
            using var wakeupService = audioManager.GetWakeupServiceInstance();

            int hitCount = 0;
            wakeupService.OnWakeup += (_, args) =>
            {
                hitCount++;
                Console.WriteLine(
                    $">>> [#{hitCount}] 唤醒命中: keyword={args.Keyword} score={args.Score}% threshold={args.Threshold}% t={args.StartTimeMs}ms");
            };

            recordService.OnDataAvailable += (audio) =>
            {
                wakeupService.FeedAudio(audio);
            };

            var inputs = recordService.ListAudioDevices(AudioDeviceType.Input)?.OrderBy(s => s.Id).ToList();
            Console.WriteLine("可用输入设备:");
            foreach (var d in inputs ?? new())
            {
                Console.WriteLine($"  {d.Id}. {d.Name}{(d.IsDefault ? "【默认】" : string.Empty)}");
            }
            Console.WriteLine();

            await recordService.StartRecordingAsync();
            Console.WriteLine("开始监听唤醒词 'Alexa'。请对着麦克风说出唤醒词, 按 Q / ESC 退出。");

            while (true)
            {
                var key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.Escape || key.Key == ConsoleKey.Q)
                {
                    Console.WriteLine("正在停止录音...");
                    await recordService.StopRecordingAsync();
                    break;
                }
            }
        }
    }
}
