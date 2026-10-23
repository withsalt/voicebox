using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.APM.WebRtc;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Wakeup.FsmnKws;
using VoiceBox.Wakeup.FsmnKws.Models;

namespace FsmnKwsDemo
{
    /// <summary>
    /// 演示 FsmnKws (FSMN+CTC) 流式唤醒。
    /// 用法:
    ///   - 无参数:                            麦克风音频经 WebRTC APM 处理后, 持续侦听模型自带的唤醒词
    ///   - 参数为 wav 文件路径 (16kHz/mono):  从 wav 推流解码
    /// 唤醒词由模型目录的 kws_meta.json 决定 (不再硬编码)。
    /// 模型资源目录默认取 BASE_MODEL_DIR/kws/kws-xiaomu-onnx, 可用环境变量 FSMN_KWS_MODEL_DIR 覆盖。
    /// </summary>
    internal class Program
    {
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";

        private static readonly string ModelDir = Path.Combine(BASE_MODEL_DIR, "kws", "kws-xiaoai-onnx");

        /// <summary>从模型目录的 kws_meta.json 读取唤醒词 (仅用于提示显示)。</summary>
        private static string ReadKeyword() =>
            FsmnKwsMetadata.Load(Path.Combine(ModelDir, FsmnKwsMetadata.FileName)).Keyword;

        static async Task Main(string[] args)
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            Console.WriteLine($"模型目录: {ModelDir}");
            if (args.Length > 0 && File.Exists(args[0]))
            {
                RunFromWav(args[0]);
                return;
            }
            await RunFromMicrophoneAsync();
        }

        private static FsmnKwsOptions BuildOptions()
        {
            var opts = new FsmnKwsOptions
            {
                ModelPath = Path.Combine(ModelDir, "kws_streaming.onnx"),
                MetaFilePath = Path.Combine(ModelDir, "kws_meta.json"),
                Threshold = 0.2f,
                DebounceMs = 800,
                ExecutionProvider = OnnxExecutionProvider.Cpu,
                ThreadsNum = 1, // ~750K 参数小模型, 单线程已远超实时
            };
            // 唤醒词由 ModelDir/kws_meta.json 自动读取, 无需配置。
            return opts;
        }

        // ----------------- File mode -----------------
        private static void RunFromWav(string wavPath)
        {
            Console.WriteLine($"Loading wav: {wavPath}");
            (float[] samples, int sampleRate) = ReadWav16kMono(wavPath);
            Console.WriteLine($"Samples={samples.Length}, rate={sampleRate}Hz, duration={samples.Length / (float)sampleRate:F2}s");
            if (sampleRate != 16000)
            {
                Console.WriteLine("ERROR: wav must be 16kHz/mono");
                return;
            }

            using var lf = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Debug);
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
            });

            using var service = new FsmnKwsService(Options.Create(BuildOptions()), lf.CreateLogger<FsmnKwsService>());

            int hits = 0;
            service.OnWakeup += (_, e) =>
            {
                hits++;
                Console.WriteLine($">>> [#{hits}] 命中: keyword={e.Keyword} score={e.Score}% threshold={e.Threshold}% t={e.StartTimeMs}ms duration={e.DurationMs}ms");
            };

            service.Start();
            const int chunkSamples = 1280; // 80ms
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int offset = 0; offset < samples.Length; offset += chunkSamples)
            {
                int take = Math.Min(chunkSamples, samples.Length - offset);
                service.FeedAudio(samples.AsSpan(offset, take));
            }
            // 尾部静音, 冲掉 lookahead 延迟 (8 帧 × 30ms) 并留出确认余量
            service.FeedAudio(new float[16000 / 2]);
            sw.Stop();

            double audioSec = (samples.Length + 16000 / 2) / 16000.0;
            Console.WriteLine($"Done. Total hits = {hits}");
            Console.WriteLine($"Processed {audioSec:F2}s audio in {sw.Elapsed.TotalMilliseconds:F1}ms " +
                              $"(RTF={sw.Elapsed.TotalSeconds / audioSec:F4}, {audioSec / sw.Elapsed.TotalSeconds:F0}x realtime)");
        }

        // ----------------- Microphone mode -----------------
        private static async Task RunFromMicrophoneAsync()
        {
            using var audioManager = new AudioServiceManager()
                .AddLogger(b =>
                {
                    b.SetMinimumLevel(LogLevel.Debug);
                    b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
                })
                .AddPortAudioDuplexService(options =>
                {
                    options.SampleRate = 16000;
                    options.ChannelCount = 1;
                    options.SampleFormat = SampleFormat.Float32;
                    options.EnableAec = true;
                    options.EnableDnr = false;
                    options.EnableHpf = false;
                })
                .AddWebRtcApmService(o =>
                {
                    o.SampleRate = 16000;
                    o.ChannelCount = 1;
                    o.EnableHighPassFilter = true;
                    o.EnableEchoCanceller = true;
                    o.EnableNoiseSuppression = true;
                    o.EnableAgc2 = true;
                })
                .AddFsmnKwsService(o =>
                {
                    o.SampleRate = 16000;
                    o.ChannelCount = 1;
                    o.ModelPath = Path.Combine(ModelDir, "kws_streaming.onnx");
                    o.MetaFilePath = Path.Combine(ModelDir, "kws_meta.json");
                    o.Threshold = 0.25f;
                    o.DebounceMs = 800;
                    o.ExecutionProvider = OnnxExecutionProvider.Cpu;
                })
                .Build();

            using var record = audioManager.GetAudioRecordInstance();
            using var wake = audioManager.GetWakeupServiceInstance();

            int hits = 0;
            wake.OnWakeup += (_, e) =>
            {
                hits++;
                Console.WriteLine($">>> [#{hits}] 命中: keyword={e.Keyword} score={e.Score}% threshold={e.Threshold}% t={e.StartTimeMs}ms");
            };

            // 全双工服务派发的是经过 WebRTC APM 处理后的音频。
            record.OnDataAvailable += audio => wake.FeedAudio(audio);

            var inputs = record.ListAudioDevices(AudioDeviceType.Input)?.OrderBy(d => d.Id).ToList();
            Console.WriteLine("可用输入设备:");
            foreach (var d in inputs ?? new()) Console.WriteLine($"  {d.Id}. {d.Name}{(d.IsDefault ? " 【默认】" : "")}");
            Console.WriteLine();

            wake.Start();
            await record.StartRecordingAsync();
            Console.WriteLine($"开始监听唤醒词 \"{ReadKeyword()}\" 。按 Q / ESC 退出。");

            while (true)
            {
                var key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.Escape || key.Key == ConsoleKey.Q)
                {
                    Console.WriteLine("停止录音...");
                    await record.StopRecordingAsync();
                    break;
                }
            }
        }

        // ----------------- WAV reader (16-bit PCM only) -----------------
        private static (float[] samples, int sampleRate) ReadWav16kMono(string path)
        {
            using var br = new BinaryReader(File.OpenRead(path));
            string riff = new string(br.ReadChars(4));
            if (riff != "RIFF") throw new InvalidDataException("Not a RIFF wav file");
            br.ReadInt32();
            string wave = new string(br.ReadChars(4));
            if (wave != "WAVE") throw new InvalidDataException("Not a WAVE wav file");

            short channels = 1, bitsPerSample = 16;
            int sampleRate = 16000;
            int dataLen = 0;
            long dataOffset = 0;

            while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
            {
                string chunkId = new string(br.ReadChars(4));
                int chunkSize = br.ReadInt32();
                if (chunkSize < 0) throw new InvalidDataException($"Invalid wav chunk size: {chunkSize}");

                if (chunkId == "fmt ")
                {
                    if (chunkSize < 16) throw new InvalidDataException("Invalid fmt chunk.");
                    short audioFormat = br.ReadInt16();
                    channels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32();
                    br.ReadInt16();
                    bitsPerSample = br.ReadInt16();
                    SkipChunkBytes(br, (long)chunkSize - 16 + (chunkSize & 1));
                    if (audioFormat != 1) throw new NotSupportedException("Only PCM wav supported.");
                }
                else if (chunkId == "data")
                {
                    dataLen = chunkSize;
                    dataOffset = br.BaseStream.Position;
                    SkipChunkBytes(br, (long)chunkSize + (chunkSize & 1));
                }
                else
                {
                    SkipChunkBytes(br, (long)chunkSize + (chunkSize & 1));
                }
            }

            if (dataOffset == 0) throw new InvalidDataException("No data chunk found");
            if (bitsPerSample != 16) throw new NotSupportedException("Only 16-bit PCM supported.");

            br.BaseStream.Seek(dataOffset, SeekOrigin.Begin);
            int frameCount = dataLen / 2 / channels;
            var samples = new float[frameCount];
            for (int i = 0; i < frameCount; i++)
            {
                int acc = 0;
                for (int c = 0; c < channels; c++) acc += br.ReadInt16();
                samples[i] = acc / (float)channels / 32768f;
            }
            return (samples, sampleRate);
        }

        private static void SkipChunkBytes(BinaryReader br, long byteCount)
        {
            if (byteCount <= 0) return;
            if (br.BaseStream.Position + byteCount > br.BaseStream.Length)
                throw new InvalidDataException("Wav chunk extends past end of file.");
            br.BaseStream.Seek(byteCount, SeekOrigin.Current);
        }
    }
}
