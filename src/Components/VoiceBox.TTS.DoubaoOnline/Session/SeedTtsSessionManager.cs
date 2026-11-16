using System.Collections.Concurrent;

namespace VoiceBox.TTS.DoubaoOnline.Session
{
    /// <summary>
    /// 活动会话注册表：将每个 <see cref="SeedTtsSession"/> 与其后台
    /// <see cref="SeedTtsConnectionWorker"/> 一并跟踪。线程安全。
    /// </summary>
    internal sealed class SeedTtsSessionManager : IDisposable
    {
        private readonly ConcurrentDictionary<string, SessionContext> _sessions = new();
        private int _disposed;

        public int Count => _sessions.Count;

        public bool TryAdd(SeedTtsSession session, SeedTtsConnectionWorker worker)
        {
            EnsureNotDisposed();
            return _sessions.TryAdd(session.SessionId, new SessionContext(session, worker));
        }

        public bool TryRemove(string sessionId, out SeedTtsSession session, out SeedTtsConnectionWorker worker)
        {
            if (_sessions.TryRemove(sessionId, out var ctx))
            {
                session = ctx.Session;
                worker = ctx.Worker;
                return true;
            }
            session = null!;
            worker = null!;
            return false;
        }

        public IReadOnlyList<(SeedTtsSession Session, SeedTtsConnectionWorker Worker)> Snapshot()
            => _sessions.Values.Select(c => (c.Session, c.Worker)).ToArray();

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            foreach (var ctx in _sessions.Values)
            {
                try { ctx.Session.Cancel(); } catch { }
                try { ctx.Session.Dispose(); } catch { }
            }
            _sessions.Clear();
        }

        private void EnsureNotDisposed()
            => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        private readonly record struct SessionContext(SeedTtsSession Session, SeedTtsConnectionWorker Worker);
    }
}
