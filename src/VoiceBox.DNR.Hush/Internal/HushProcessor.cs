using VoiceBox.Core.Onnx;

namespace VoiceBox.DNR.Hush.Internal
{
    /// <summary>
    /// Hush 16 kHz 逐帧处理引擎，端口自 deepfilter-rt 的 <c>DeepFilterProcessor.process_frame</c>
    /// （combined_streaming / StatefulH0 路径）。每次处理一个 hop（160 样本 @16k），输出一个 hop。
    ///
    /// 流水线：STFT → ERB/Spec 特征(运行归一化) → 有状态 ONNX → ERB 掩码(lsnr 阈值) → 深度滤波 → ISTFT。
    /// 编码器卷积上下文用一个 encWindow 帧的环形缓冲（预填零提供 warmup）；ONNX 内部切到末帧进 GRU。
    /// 非线程安全：由上层服务持锁串行调用。
    /// </summary>
    internal sealed class HushProcessor : IDisposable
    {
        // Hush 的 lsnr 区间为 (-15, 35)（lsnr_min=-15, lsnr_max=35），下列阈值与官方 Python 推理
        // 等价（恒不触发门控，掩码与深度滤波始终施加），与 infer_single 的无条件路径一致。
        private const float MinDbThresh = -15f;
        private const float MaxDbErbThresh = 35f;
        private const float MaxDbDfThresh = 35f;

        private readonly DfState _df;
        private readonly HushOnnxRunner _runner;
        private readonly float _alpha;
        private readonly int _freqSize;
        private readonly int _nbErb;
        private readonly int _nbDf;
        private readonly int _dfOrder;
        private readonly int _lookahead;
        private readonly int _encWindow;

        // 当前帧谱与特征工作缓冲。
        private readonly float[] _workRe;       // [freqSize]
        private readonly float[] _workIm;       // [freqSize]
        private readonly float[] _featErbFrame; // [nbErb]
        private readonly float[] _featSpecRe;   // [nbDf]
        private readonly float[] _featSpecIm;   // [nbDf]
        private readonly float[] _mask;         // [nbErb]
        private readonly float[] _zeros;        // [nbErb]
        private readonly float[] _coefs;        // [nbDf*dfOrder*2]
        private readonly float[] _outRe;        // [freqSize]
        private readonly float[] _outIm;        // [freqSize]

        // 编码器上下文环形缓冲（slot-major）。
        private readonly float[] _encErbRing;   // [encWindow*nbErb]
        private readonly float[] _encSpecRing;  // [encWindow*nbDf*2]，每频点交错 re/im
        private long _encRingPos;
        private int _encRingCount;

        // 推理输入扁平缓冲。
        private readonly float[] _infErb;       // [encWindow*nbErb]
        private readonly float[] _infSpec;      // [2*encWindow*nbDf]，planar

        // 滚动谱缓冲：X 为深度滤波用的带噪谱，Y 为输出/掩码用的谱。oldest-first，frame-major。
        private readonly int _bufXLen;
        private readonly int _bufYLen;
        private readonly float[] _xRe;          // [bufXLen*freqSize]
        private readonly float[] _xIm;
        private readonly float[] _yRe;          // [bufYLen*freqSize]
        private readonly float[] _yIm;

        private float _lastLsnr;

        public int HopSize => _df.FrameSize;
        public int EncWindow => _encWindow;
        public int Lookahead => _lookahead;
        public int FreqSize => _freqSize;
        public float NormAlpha => _alpha;

        // 测试用：暴露最近一帧的中间量，便于与 Python 参考逐级对拍（InternalsVisibleTo）。
        internal ReadOnlySpan<float> LastFeatErb => _featErbFrame;
        internal ReadOnlySpan<float> LastFeatSpecRe => _featSpecRe;
        internal ReadOnlySpan<float> LastFeatSpecIm => _featSpecIm;
        internal ReadOnlySpan<float> LastMask => _mask;
        internal ReadOnlySpan<float> LastCoefs => _coefs;
        internal float LastLsnr => _lastLsnr;

        public HushProcessor(
            string modelPath,
            OnnxExecutionOptions onnxOptions,
            int sr, int fftSize, int hopSize, int nbErb, int nbDf, int dfOrder,
            int lookahead, int minNbErbFreqs, float normTau, int encWindow)
        {
            if (dfOrder <= 0)
                throw new ArgumentOutOfRangeException(nameof(dfOrder), dfOrder, "dfOrder must be > 0.");
            if (lookahead < 0)
                throw new ArgumentOutOfRangeException(nameof(lookahead), lookahead, "lookahead must be >= 0.");

            _df = new DfState(sr, fftSize, hopSize, nbErb, nbDf, minNbErbFreqs);
            _alpha = DfState.ComputeNormAlpha(sr, hopSize, normTau);

            HushOnnxRunner? runner = null;
            try
            {
                runner = new HushOnnxRunner(modelPath, onnxOptions, encWindow);

                // 以模型张量维度为准校验配置一致性。
                if (runner.NbErb != nbErb)
                    throw new InvalidOperationException($"config nb_erb={nbErb} != model nb_erb={runner.NbErb}.");
                if (runner.NbDf != nbDf)
                    throw new InvalidOperationException($"config nb_df={nbDf} != model nb_df={runner.NbDf}.");
                if (runner.DfOrder != dfOrder)
                    throw new InvalidOperationException($"config df_order={dfOrder} != model df_order={runner.DfOrder}.");

                _runner = runner;
                runner = null;
            }
            finally
            {
                runner?.Dispose();
            }

            _freqSize = _df.FreqSize;
            _nbErb = nbErb;
            _nbDf = nbDf;
            _dfOrder = dfOrder;
            _lookahead = lookahead;
            _encWindow = encWindow;

            _workRe = new float[_freqSize];
            _workIm = new float[_freqSize];
            _featErbFrame = new float[nbErb];
            _featSpecRe = new float[nbDf];
            _featSpecIm = new float[nbDf];
            _mask = new float[nbErb];
            _zeros = new float[nbErb];
            _coefs = new float[nbDf * dfOrder * 2];
            _outRe = new float[_freqSize];
            _outIm = new float[_freqSize];

            _encErbRing = new float[encWindow * nbErb];
            _encSpecRing = new float[encWindow * nbDf * 2];

            _infErb = new float[encWindow * nbErb];
            _infSpec = new float[2 * encWindow * nbDf];

            _bufXLen = Math.Max(dfOrder, Math.Max(lookahead, 1));
            _bufYLen = dfOrder + lookahead;
            _xRe = new float[_bufXLen * _freqSize];
            _xIm = new float[_bufXLen * _freqSize];
            _yRe = new float[_bufYLen * _freqSize];
            _yIm = new float[_bufYLen * _freqSize];

            ResetRings();
        }

        private void ResetRings()
        {
            // enc 环形预填零并把 pos/count 置为 encWindow-1（首个真实帧即填满，warmup 由零帧提供）。
            Array.Clear(_encErbRing);
            Array.Clear(_encSpecRing);
            _encRingPos = _encWindow - 1;
            _encRingCount = _encWindow - 1;
            Array.Clear(_xRe);
            Array.Clear(_xIm);
            Array.Clear(_yRe);
            Array.Clear(_yIm);
        }

        /// <summary>复位全部状态，开始新的独立流。</summary>
        public void Reset()
        {
            _df.Reset();
            _runner.Reset();
            ResetRings();
        }

        /// <summary>处理一个 hop（<see cref="HopSize"/> 个 16k 样本），输出一个 hop。</summary>
        public void ProcessFrame(ReadOnlySpan<float> input, Span<float> output)
        {
            // 1. STFT。
            _df.Analysis(input, _workRe, _workIm);

            // 2. 把原始谱压入滚动缓冲 X / Y（深度滤波用 X，输出/掩码用 Y）。
            PushFrame(_xRe, _xIm, _bufXLen);
            PushFrame(_yRe, _yIm, _bufYLen);

            // 3-4. ERB / 复数谱特征（运行归一化）。
            _df.FeatErb(_workRe, _workIm, _alpha, _featErbFrame);
            _df.FeatCplx(_workRe, _workIm, _alpha, _featSpecRe, _featSpecIm);

            // 5. 写入编码器上下文环形缓冲（direct features，无 lookahead 延迟）。
            int slot = (int)(_encRingPos % _encWindow);
            Array.Copy(_featErbFrame, 0, _encErbRing, slot * _nbErb, _nbErb);
            int specOff = slot * _nbDf * 2;
            for (int f = 0; f < _nbDf; f++)
            {
                _encSpecRing[specOff + f * 2] = _featSpecRe[f];
                _encSpecRing[specOff + f * 2 + 1] = _featSpecIm[f];
            }
            _encRingPos++;
            _encRingCount = Math.Min(_encRingCount + 1, _encWindow);

            // 6. warmup 不足则输出静音（StatefulH0 的 min_required=1，通常首帧即满足）。
            if (_encRingCount < 1)
            {
                Array.Clear(_outRe);
                Array.Clear(_outIm);
                _df.Synthesis(_outRe, _outIm, output);
                return;
            }

            // 7. 按 oldest-first 组装推理输入张量。
            int t = _encRingCount; // = encWindow（首帧后恒定）
            int oldest = (int)(((_encRingPos - t) % _encWindow + _encWindow) % _encWindow);
            FillInferenceTensors(t, oldest);

            // 8. 推理。
            float lsnr = _runner.Run(_infErb, _infSpec, _mask, _coefs);
            _lastLsnr = lsnr;

            // 9. ERB 掩码：对 Y[mask_idx] 原地施加。
            int maskIdx = _bufYLen - 1 - _lookahead;
            bool applyMask = lsnr <= MaxDbErbThresh;
            bool applyZeroMask = lsnr < MinDbThresh;
            if (applyMask)
            {
                Span<float> yRe = _yRe.AsSpan(maskIdx * _freqSize, _freqSize);
                Span<float> yIm = _yIm.AsSpan(maskIdx * _freqSize, _freqSize);
                _df.ApplyMask(yRe, yIm, applyZeroMask ? _zeros : _mask);
            }

            // 10. 取出该帧作为输出基底。
            _yRe.AsSpan(maskIdx * _freqSize, _freqSize).CopyTo(_outRe);
            _yIm.AsSpan(maskIdx * _freqSize, _freqSize).CopyTo(_outIm);

            // 11. 深度滤波覆盖前 nbDf 个频点（用 X 的最旧 dfOrder 帧）。
            bool applyDf = lsnr <= MaxDbDfThresh && !applyZeroMask;
            if (applyDf)
            {
                DeepFilterOp.Apply(
                    _outRe, _outIm,
                    _xRe.AsSpan(0, _dfOrder * _freqSize),
                    _xIm.AsSpan(0, _dfOrder * _freqSize),
                    _coefs, _nbDf, _dfOrder, _freqSize);
            }

            // 12. ISTFT。
            _df.Synthesis(_outRe, _outIm, output);
        }

        /// <summary>把当前帧谱 push 进一个 oldest-first 滚动缓冲（左移一帧 + 末尾写当前帧）。</summary>
        private void PushFrame(float[] re, float[] im, int len)
        {
            if (len > 1)
            {
                Array.Copy(re, _freqSize, re, 0, (len - 1) * _freqSize);
                Array.Copy(im, _freqSize, im, 0, (len - 1) * _freqSize);
            }
            int last = (len - 1) * _freqSize;
            _workRe.AsSpan(0, _freqSize).CopyTo(re.AsSpan(last, _freqSize));
            _workIm.AsSpan(0, _freqSize).CopyTo(im.AsSpan(last, _freqSize));
        }

        /// <summary>从 enc 环形缓冲按 oldest-first 组装 [1,1,T,nbErb] 与 [1,2,T,nbDf] 扁平数据。</summary>
        private void FillInferenceTensors(int t, int oldestSlot)
        {
            for (int ti = 0; ti < t; ti++)
            {
                int s = (oldestSlot + ti) % _encWindow;
                Array.Copy(_encErbRing, s * _nbErb, _infErb, ti * _nbErb, _nbErb);
            }

            int ch1Offset = t * _nbDf; // 虚部通道起点（planar）
            for (int ti = 0; ti < t; ti++)
            {
                int s = (oldestSlot + ti) % _encWindow;
                int srcOff = s * _nbDf * 2;
                int reOff = ti * _nbDf;
                int imOff = ch1Offset + ti * _nbDf;
                for (int f = 0; f < _nbDf; f++)
                {
                    _infSpec[reOff + f] = _encSpecRing[srcOff + f * 2];
                    _infSpec[imOff + f] = _encSpecRing[srcOff + f * 2 + 1];
                }
            }
        }

        public void Dispose() => _runner.Dispose();
    }
}
