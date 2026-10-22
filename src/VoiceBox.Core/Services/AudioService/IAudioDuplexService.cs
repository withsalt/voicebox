using System;
using System.Collections.Generic;
using System.Text;

using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Core.Services.AudioService
{
    public interface IAudioDuplexService : IAudioRecordService, IAudioPlayService
    {
        /// <summary>
        /// 全双工音频回调中的多路音频帧快照。事件由音频服务后台线程分发，包含 render 参考、
        /// 原始采集、AEC 前采集和 AEC 后采集。事件参数中的内存仅在回调期间有效，需长期保存时应复制。
        /// </summary>
        event DuplexAudioFrameHandler? OnDuplexAudioFrame;

        /// <summary>
        /// 录音处理链（当前为可选 DNR）实际产出的音频帧。SourceRange 精确指向原始采集时间轴，
        /// 因而消费者不得再用“最近一次说话人结果”代替同区间证据。事件内存只在回调期间有效。
        /// </summary>
        event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame;

        /// <summary>
        /// 启动全双工音频流（同时录音和播放）。
        /// </summary>
        /// <param name="inputDeviceIndex">输入设备索引，-1 表示使用默认设备。</param>
        /// <param name="outputDeviceIndex">输出设备索引，-1 表示使用默认设备。</param>
        /// <param name="cancellationToken">取消令牌。</param>
        Task StartAsync(int inputDeviceIndex = -1, int outputDeviceIndex = -1, CancellationToken cancellationToken = default);

        /// <summary>
        /// 停止全双工音频流。
        /// </summary>
        /// <param name="cancellationToken">取消令牌。</param>
        Task StopAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// 采集处理器开关（运行时可切换，A/B 对比）。跨线程仅作 bool 读写。
        /// </summary>
        bool EnableAec { get; set; }

        /// <summary>
        /// 降噪处理器开关（运行时可切换，A/B 对比）。跨线程仅作 bool 读写。
        /// </summary>
        bool EnableDnr { get; set; }

        /// <summary>
        /// 高通滤波处理器开关（运行时可切换，A/B 对比）。跨线程仅作 bool 读写。
        /// </summary>
        bool EnableHighPassFilter { get; set; }
    }
}
