namespace VoiceBox.Assistant.Runtime.Gates
{
    /// <summary>
    /// 麦克风输入门。需要串行识别时关闭, 让处理后采集帧的识别管线短路;
    /// 启用可打断回复时, Thinking/Speaking 阶段会重新打开。
    /// </summary>
    internal sealed class InputGate
    {
        private int _open = 1;

        public bool IsOpen => Volatile.Read(ref _open) == 1;

        public void Open() => Interlocked.Exchange(ref _open, 1);

        public void Close() => Interlocked.Exchange(ref _open, 0);
    }
}
