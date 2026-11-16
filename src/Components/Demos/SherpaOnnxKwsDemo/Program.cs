using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DNR.DPDFNet;
using VoiceBox.DNR.DPDFNet.Models;
using VoiceBox.Wakeup.SherpaOnnxKws;
using VoiceBox.Wakeup.SherpaOnnxKws.Models;

namespace SherpaOnnxKwsDemo
{
    /// <summary>
    /// 演示 SherpaOnnxKws 流式命令词唤醒。
    /// 用法:
    ///   - 无参数:                            读取麦克风, 持续侦听
    ///   - 参数为 wav 文件路径 (16kHz/mono):  从 wav 推流解码
    /// </summary>
    internal class Program
    {
        // 模型根目录 (可通过环境变量 KWS_MODEL_DIR 覆盖)
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";

        private static readonly string ModelDir =
            Environment.GetEnvironmentVariable("KWS_MODEL_DIR")
            ?? Path.Combine(BASE_MODEL_DIR, "kws", "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01");

        static async Task Main(string[] args)
        {
            if (OperatingSystem.IsWindows())
            {
                // 设备名 (例如 "Intel® Smart Sound...") 含非 GBK 字符, 默认控制台编码会显示成 '?'
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            if (args.Length > 0 && File.Exists(args[0]))
            {
                RunFromWav(args[0]);
                return;
            }
            await RunFromMicrophoneAsync();
        }

        private static SherpaOnnxKwsOptions BuildOptions()
        {
            var opts = new SherpaOnnxKwsOptions
            {
                SampleRate = 16000,
                ChannelCount = 1,
                EncoderModelPath = Path.Combine(ModelDir, "encoder-epoch-99-avg-1-chunk-16-left-64.onnx"),
                DecoderModelPath = Path.Combine(ModelDir, "decoder-epoch-99-avg-1-chunk-16-left-64.onnx"),
                JoinerModelPath = Path.Combine(ModelDir, "joiner-epoch-99-avg-1-chunk-16-left-64.onnx"),
                TokensFilePath = Path.Combine(ModelDir, "tokens.txt"),
                // 对齐 sherpa-onnx 官方 zh-en KWS 推荐参数:
                //   keywords-score=2.0, keywords-threshold=0.25, num-active-paths=8.
                // 阈值过低 (0.1) 会让 beam 搜索倾向选择常见声学路径而非关键词路径,
                // 反而降低命中率; boost 偏低 (1.5) 会让关键词前缀过早被剔出 beam。
                KeywordsThreshold = 0.20f,
                KeywordsScore = 3.0f,
                MaxActivePaths = 8,
                NumTrailingBlanks = 1,
                DebounceMs = 800,
                ConfusionVariants = KwsConfusionVariants.None,
                ExecutionProvider = OnnxExecutionProvider.Cpu,
                ThreadsNum = 2,
            };

            // 使用 AddKeywordWithVariants 注册「主关键词 + 发音变体」: 命中任一变体都上报 "叮当叮当"。
            // 弥补 3M 小模型对 4 字命令置信度低、常把声调/相近韵母解错的问题。
            opts.AddKeywordWithVariants(
                keyword: "叮当叮当",
                variants: new[]
                {
                    "叮当叮", "当叮当", "丁当丁当", "顶当顶当", "定当定当", "叮挡叮挡", "叮荡叮荡",
                    "顶挡顶挡", "定荡定荡", "叮当定当", "定当叮当", "叮单叮单", "叮当叮荡",
                });

            return opts;
        }

        // ----------------- File mode -----------------
        private static void RunFromWav(string wavPath)
        {
            Console.WriteLine($"Loading wav: {wavPath}");
            (float[] samples, int sampleRate) = ReadWav16kMono(wavPath);
            Console.WriteLine($"Samples={samples.Length}, sample rate={sampleRate}Hz, duration={samples.Length / (float)sampleRate:F2}s");
            if (sampleRate != 16000)
            {
                Console.WriteLine("ERROR: wav must be 16kHz/mono");
                return;
            }

            var opts = BuildOptions();
            using var lf = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Debug);
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
            });

            var ioptions = Options.Create(opts);
            using var service = new SherpaOnnxKwsService(ioptions, lf.CreateLogger<SherpaOnnxKwsService>());

            int hits = 0;
            service.OnWakeup += (_, e) =>
            {
                hits++;
                Console.WriteLine($">>> [#{hits}] 命中: keyword={e.Keyword} score={e.Score}% threshold={e.Threshold}% t={e.StartTimeMs}ms duration={e.DurationMs}ms");
            };

            service.Start();
            // 模拟流式: 每次喂 80ms (1280 sample) 数据
            const int chunkSamples = 1280;
            int offset = 0;
            while (offset < samples.Length)
            {
                int take = Math.Min(chunkSamples, samples.Length - offset);
                service.FeedAudio(samples.AsSpan(offset, take));
                offset += take;
            }
            // 尾部静音 (0.66s, 与 sherpa-onnx Python 例子保持一致), 帮助触发尾帧
            float[] tail = new float[(int)(0.66f * sampleRate)];
            service.FeedAudio(tail);

            Console.WriteLine($"Done. Total hits = {hits}");
        }

        // ----------------- Microphone mode -----------------
        private static async Task RunFromMicrophoneAsync()
        {
            using var audioManager = new AudioServiceManager()
                .AddLogger(b =>
                {
                    // Debug 级别可看到 "near-miss" 日志, 帮助调阈值
                    b.SetMinimumLevel(LogLevel.Debug);
                    b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
                })
                .AddPortAudioRecordService(options =>
                {
                    options.SampleRate = 16000;
                    options.ChannelCount = 1;
                    options.SampleFormat = SampleFormat.Float32;
                })
                .AddDPDFNetDNRService(options =>
                {
                    options.SampleRate = 16000;
                    options.ChannelCount = 1;
                    options.SampleFormat = SampleFormat.Float32;
                    options.Model = DPDFNetModel.DpDfNet8;
                    options.ExecutionProvider = OnnxExecutionProvider.Cpu;
                    options.AttenuationLimitDb = null; // 完全采用模型输出
                })
                .AddSherpaOnnxKwsService(o =>
                {
                    var template = BuildOptions();
                    o.EncoderModelPath = template.EncoderModelPath;
                    o.DecoderModelPath = template.DecoderModelPath;
                    o.JoinerModelPath = template.JoinerModelPath;
                    o.TokensFilePath = template.TokensFilePath;
                    o.KeywordsThreshold = template.KeywordsThreshold;
                    o.KeywordsScore = template.KeywordsScore;
                    o.MaxActivePaths = template.MaxActivePaths;
                    o.NumTrailingBlanks = template.NumTrailingBlanks;
                    o.DebounceMs = template.DebounceMs;
                    foreach (var k in template.Keywords) o.Keywords.Add(k);
                    o.ExecutionProvider = template.ExecutionProvider;
                    o.ThreadsNum = template.ThreadsNum;
                    o.EmitImmediately = true;
                    o.ConfusionVariants = template.ConfusionVariants;
                })
                .Build();

            using var record = audioManager.GetAudioRecordInstance();
            using var dnr = audioManager.GetDNRServiceInstance();
            using var wake = audioManager.GetWakeupServiceInstance();

            int hits = 0;
            wake.OnWakeup += (_, e) =>
            {
                hits++;
                Console.WriteLine($">>> [#{hits}] 命中: keyword={e.Keyword} score={e.Score}% threshold={e.Threshold}% t={e.StartTimeMs}ms");
            };

            // 音频管道: 录音 → 降噪 → 唤醒词识别
            float[] audioBuffer = new float[16000]; // 1秒缓冲区
            record.OnDataAvailable += audio =>
            {
                // 确保缓冲区足够大
                if (audio.Length > audioBuffer.Length)
                {
                    audioBuffer = new float[audio.Length];
                }

                // 复制到可写缓冲区
                audio.CopyTo(audioBuffer);

                int denoised = dnr.Process(audioBuffer.AsSpan(0, audio.Length));
                if (denoised > 0)
                {
                    wake.FeedAudio(audioBuffer.AsSpan(0, denoised));
                }
            };

            var inputs = record.ListAudioDevices(AudioDeviceType.Input)?.OrderBy(d => d.Id).ToList();
            Console.WriteLine("可用输入设备:");
            foreach (var d in inputs ?? new()) Console.WriteLine($"  {d.Id}. {d.Name}{(d.IsDefault ? " 【默认】" : "")}");
            Console.WriteLine();

            await record.StartRecordingAsync();
            Console.WriteLine($"模型目录: {ModelDir}");
            Console.WriteLine("降噪: DPDFNet8 (CPU)");
            Console.WriteLine("开始监听唤醒词。请对着麦克风说出 keywords.txt 中配置的关键词 (例如 '叮当叮当' 等)。按 Q / ESC 退出。");

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
            // RIFF header
            string riff = new string(br.ReadChars(4));
            if (riff != "RIFF") throw new InvalidDataException("Not a RIFF wav file");
            br.ReadInt32(); // file size
            string wave = new string(br.ReadChars(4));
            if (wave != "WAVE") throw new InvalidDataException("Not a WAVE wav file");

            short channels = 1, bitsPerSample = 16;
            int sampleRate = 16000;
            int dataLen = 0;
            long dataOffset = 0;

            while (br.BaseStream.Position < br.BaseStream.Length)
            {
                string chunkId = new string(br.ReadChars(4));
                int chunkSize = br.ReadInt32();
                if (chunkId == "fmt ")
                {
                    short audioFormat = br.ReadInt16();
                    channels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32(); // byteRate
                    br.ReadInt16(); // blockAlign
                    bitsPerSample = br.ReadInt16();
                    if (chunkSize > 16) br.BaseStream.Seek(chunkSize - 16, SeekOrigin.Current);
                    if (audioFormat != 1) throw new NotSupportedException("Only PCM wav supported.");
                }
                else if (chunkId == "data")
                {
                    dataLen = chunkSize;
                    dataOffset = br.BaseStream.Position;
                    br.BaseStream.Seek(chunkSize, SeekOrigin.Current);
                }
                else
                {
                    br.BaseStream.Seek(chunkSize, SeekOrigin.Current);
                }
            }

            if (channels != 1) throw new NotSupportedException("Only mono wav supported.");
            if (bitsPerSample != 16) throw new NotSupportedException("Only 16-bit PCM supported.");

            br.BaseStream.Seek(dataOffset, SeekOrigin.Begin);
            int numSamples = dataLen / 2;
            var samples = new float[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                short s = br.ReadInt16();
                samples[i] = s / 32768f;
            }
            return (samples, sampleRate);
        }
    }
}
