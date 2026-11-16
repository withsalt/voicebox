namespace VoiceBox.Audio.PortAudio.Backend;

/// <summary>
/// PortAudio 单例管理器
/// </summary>
internal sealed class PortAudioManager : IDisposable
{
    private readonly object _lock = new();
    private bool _isInitialized = false;
    private bool _isDisposed = false;
    private int _referenceCount = 0;

    public static readonly PortAudioManager Instance = new PortAudioManager();

    private PortAudioManager()
    {
        // Private constructor for singleton
    }

    /// <summary>
    /// 获取并增加引用计数，确保 PortAudio 已初始化
    /// </summary>
    public bool AcquireReference()
    {
        lock (_lock)
        {
            if (_isDisposed)
            {
                return false;
            }

            if (!_isInitialized)
            {
                PortAudioBackend.Initialize();
                _isInitialized = true;
            }

            _referenceCount++;
            return true;
        }
    }

    /// <summary>
    /// 释放引用计数，当引用计数为 0 时终止 PortAudio
    /// </summary>
    public void ReleaseReference()
    {
        lock (_lock)
        {
            if (_referenceCount > 0)
            {
                _referenceCount--;
                if (_referenceCount == 0 && _isInitialized)
                {
                    PortAudioBackend.Terminate();
                    _isInitialized = false;
                }
            }
        }
    }

    /// <summary>
    /// 检查 PortAudio 是否已初始化
    /// </summary>
    public bool IsInitialized
    {
        get
        {
            lock (_lock)
            {
                return _isInitialized && !_isDisposed;
            }
        }
    }

    /// <summary>
    /// 获取当前引用计数
    /// </summary>
    public int ReferenceCount
    {
        get
        {
            lock (_lock)
            {
                return _referenceCount;
            }
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_isDisposed) return;

            try
            {
                if (_isInitialized)
                {
                    PortAudioBackend.Terminate();
                    _isInitialized = false;
                }
            }
            finally
            {
                _isDisposed = true;
                _referenceCount = 0;
            }
        }
    }
}
