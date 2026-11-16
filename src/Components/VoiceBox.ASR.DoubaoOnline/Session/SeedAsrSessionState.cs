namespace VoiceBox.ASR.DoubaoOnline.Session
{
    internal enum SeedAsrSessionState
    {
        /// <summary>接受 FeedAudio，连接与收发循环运行中。</summary>
        Active = 0,

        /// <summary>已调用 FinishSession：停止接收新音频，等待发送最后一包并排空结果。</summary>
        Finishing = 1,

        /// <summary>会话结束，连接已关闭。</summary>
        Finished = 2,
    }
}
