using System.Threading;

namespace VoiceBox.LLM.LLamaSharp.Internal
{
    internal static class LlamaCancellation
    {
        public static CancellationTokenSource? LinkIfNeeded(
            CancellationToken requestToken,
            CancellationToken shutdownToken,
            out CancellationToken token)
        {
            if (!requestToken.CanBeCanceled)
            {
                token = shutdownToken;
                return null;
            }

            if (!shutdownToken.CanBeCanceled)
            {
                token = requestToken;
                return null;
            }

            var linked = CancellationTokenSource.CreateLinkedTokenSource(requestToken, shutdownToken);
            token = linked.Token;
            return linked;
        }

        public static CancellationTokenSource CreateLinkedSource(
            CancellationToken requestToken,
            CancellationToken shutdownToken)
        {
            if (requestToken.CanBeCanceled && shutdownToken.CanBeCanceled)
                return CancellationTokenSource.CreateLinkedTokenSource(requestToken, shutdownToken);
            if (requestToken.CanBeCanceled)
                return CancellationTokenSource.CreateLinkedTokenSource(requestToken);
            if (shutdownToken.CanBeCanceled)
                return CancellationTokenSource.CreateLinkedTokenSource(shutdownToken);
            return new CancellationTokenSource();
        }
    }
}
