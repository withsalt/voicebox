# VoxCPM-0.5B ONNX 导出工作日志

## 2026-07-02 初版导出链路

### 环境
- conda env `voxcpm`: Python 3.13.14, torch 2.12.1+cu132, onnx 1.21, onnxruntime 1.26。
- 补装: transformers 5.12 / tokenizers / soundfile / einops / pydantic / onnxconverter-common / librosa / torchaudio 2.11(+cpu)。
- 本机坑 1: `conda run` 偶发崩溃 → 直接用 `E:/.packages/conda/envs/voxcpm/python.exe`。
- 本机坑 2: torch 2.12+cu132 与系统 cuDNN 子库版本不匹配, CUDA conv1d 报
  `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` → 导出/参考一律 CPU fp32;
  若需 GPU 基线, `torch.backends.cudnn.enabled=False` 可绕过(_common.load_vox 已设)。
- 本机坑 3: torchaudio 2.11 的 `load()` 需要 torchcodec → 音频解码全部走 soundfile,
  重采样用 `torchaudio.functional.resample`(纯算子, 可用)。

### 结构决策(与 qwen3-tts-onnx-export 对齐)
- **统一 prefill/decode 单图 `lm.onnx`**(base_lm 24L + FSQ mix + residual_lm 6L + dit/stop 头):
  分开导出会让 ~1.8GB 权重复制两份。fsq_mask/feat_embed_masked 输入把
  prefill(按 text/audio mask 混合)与 decode(恒 audio 语义)统一成同一图。
- 文本 embedding 留在宿主(embeddings/text_embedding.bin, 73448x1024 fp32):
  int4 量化只碰 MatMul 权重, Gather 表留 fp32 没有收益; C# 用 EmbeddingStore 查表。
- SDPA(enable_gqa) → 手写 attention(expand+flatten GQA, cumsum 因果 mask,
  softmax fp32); StaticKVCache(就地 scatter)→ 标准 past/present concat, 动态 seq 轴。
- **CFM 10 步 Euler 整体展开为单图 `cfm_decode.onnx`**(静态形状): 每 80ms patch 一次
  Run(而不是 9 次 estimator Run)。噪声 z 是图输入 → 宿主控制随机性, 跨 torch/ORT/C# 可复现。
  cfg_value 也是输入(运行时可调)。t_span/sway/zero-init(=1 步) 全部烘焙。
- feat_encoder.onnx 动态 T: prompt 一次编码 + 每步 T=1 复用同一份权重。
- AudioVAE 恒 fp32: 导出前 remove_weight_norm(62 个模块) + 用普通函数替换
  @torch.jit.script snake(3D 输入下逐元素等价)。encoder 输入要求已按 1280 左填充(宿主做)。
- LongRoPE: 序列 ≤4096 ≪ 32768 → 恒 short_factor, scaling_factor=1, 直接切表烘焙常量。
- tokenizer: `mask_multichar_chinese_tokens`(≥2 纯中文字 token 拆单字)在导出期
  预计算成 id→ids 映射表(embeddings/multichar_zh.json, 19697 项), C# 编码后查表展开,
  与 python wrapper 语义逐 id 等价。注意参考实现走 `tokenize()`——**不加 BOS**。

### 数值验证(fp32, CPU ORT vs CPU torch)
- 逐图: lm prefill/decode、feat_encoder、cfm、vae encode/decode 全部 SNR > 106dB。
- e2e(噪声注入对拍, 步数完全一致): en 80.8dB / zh 100.1dB / clone 86.7dB /
  流式 80.7dB(43/33/68 步)。
- **坑: 克隆链路的混沌性** — prompt 潜变量经 torch VAE 与 ORT VAE 有 ~1e-3 差异,
  FSQ round(x*9)/9 会把它放大成轨迹分叉(首次对拍 wav SNR 只有 2.1dB, cos 0.687,
  但两边都是合法语音)。修复: torch 参考侧也使用 ONNX 编码的 prompt 特征
  (VAE encoder 单独门控 95.6dB), 之后 86.7dB。
  ⇒ 对 C# 的启示: **凡是喂进 AR 环路的输入必须与 golden 完全一致**(prompt 特征用
  golden npy, 噪声用 golden npy), 否则只能做"合法性"级别的判断。
- KV cache fp32 主副本坑: from_local 构造期按 config.dtype(bf16) 分配 StaticKVCache,
  `.float()` 不会转 cache → 参考推理时 index_put dtype 报错 → load_vox 里重建 fp32 cache。

### 产物
- out/voxcpm-0.5b-onnx-fp32/ ≈ 2917MB(lm 1811 + fenc 244 + cfm 258 + vae 108+192 + embed 301)。
- out/voxcpm-0.5b-onnx-fp16/ ≈ 1762MB(原生半精 trace; onnxconverter 转换会因图内 fp32 Cast 报
  Add 类型冲突, 弃用)。out/voxcpm-0.5b-onnx-int4/ ≈ 1410MB(lm 1811→303MB, 削波率 ≤0.03%)。
- out/golden/: en/zh/clone/stream 的 ids/latents/wav/noise npy + prompt_16k.wav(**FLOAT
  subtype**, PCM_16 的量化差会被 AR 混沌放大到 37dB)+ prompt_feat.npy + golden_meta.json。

### C# 侧结果(src/VoiceBox.TTS.VoxCPM)
- 与 golden **逐位一致**(latents/wav/prompt_feat SNR=∞, 步数一致), tokenizer 3 案例全 MATCH。
- 15/15 单测通过; demo 无头基准: CPU fp32 首包 139ms / int4 首包 73ms RTF≈1.2。

### Code review 修复(2026-07-02)
- 4096 RoPE 表越界防护(pipeline+engine, 长输入原本中途崩溃)、badcase 重试(对齐上游默认)、
  MaxLen 护栏语义、MaxNewFrames 不再被放大、audio_start/CFM 步数单一事实来源
  (`_common.GENERATION_DEFAULTS`)、LmDecoder CPU 路径 KV 零拷贝(present OrtValue 直接回喂)、
  EmbeddingStore 按需逐行读(省 300MB 常驻)、测试常量改读包配置、文本清理去掉多余 Trim。
- 未做(记录为后续优化): 流式 VAE 解码与下一步生成的流水线重叠、session 并行加载、
  BPE 优先队列化(长文本 O(n²))、golden case 清单驱动 fixture。

## 2026-07-02 CPU 流式性能优化(效果逐比特不变)

目标: fp32 效果完全不变的前提下把 CPU 流式 RTF 压到 <1(9950X 16C/32T, 8 线程)。
基线剖析(148 步, 11.84s 音频): 每步 = cfm 36.6ms + lm 33.0ms + fe 4.6ms(AR 串行链)
+ vae 滑窗 23.9ms(与 AR 串行)= 98ms > 80ms 预算 → RTF 1.23, 播放断续。
瓶颈本质是**内存带宽**(cfm 258MB×9 评估 + lm 1.8GB 权重, 每步 ~4GB 读)。

### 改动
- **A1 流水线化**(VoxCpmPipeline.SynthesizeStream): AR 环路(生产 patch)与 VAE 滑窗
  解码(消费)无数据依赖 → BlockingCollection 双线程流水线, 步耗时 AR+VAE → max(AR,VAE)。
  patch 顺序/滑窗组成不变 ⇒ 输出逐比特一致(SHA256 对拍验证)。
- **A2 线程划分**: VAE 独立 session + 独立小线程池(固定按 max(2, ThreadsNum/2) 取)
  + 关自旋。实测主池 8 线程(=单 CCD)最优, >8 跨 CCD 剧烈退化
  (12 线程 lm 33→60ms); fe 拆独立 1/2 线程池反而更慢(微图吃并行), 保持共享大池。
- **B1** OnnxSessionFactory.CreateCpu: EXTENDED → ORT_ENABLE_ALL(NCHWc 布局, VAE conv
  收益 ~10%), 与 python ORT 默认一致; 实测输出 SHA 不变。
- **B2** session.set_denormal_as_zero 固定开启: 本内容下无 denormal, 中性,
  留作防御。
- **B5 GQA 广播化**(_modules.minicpm_layers): repeat_kv Expand 物化 → 分组广播 MatMul
  (q [B,KV,g,S,Dh] × k/v [B,KV,1,T,Dh]), 点积同序同值 ⇒ 逐比特一致, 消除随 P 线性增长的
  KV 复制(P=500 时 ~200MB/步)。lm/feat_encoder/cfm(DiT/LocEnc 亦是 16头/2KV GQA)
  三图重导出, 门禁 106-141dB。长句(494 步)lm 末段 47-49ms → 43-44ms。
- 预热改走流式路径(覆盖滑窗 VAE 1/2/3-patch 形状), 首次流式合成的冷核尖峰消除
  (run1 首包 299→138ms)。

### 结果(同文本同 seed)
- 短句(148 步): RTF 1.23 → **0.99-1.01**; 长句(494 步): 1.10 → 1.05(剩余增长来自
  present KV 全量 concat, 需 GQA 原地 buffer, 记为后续)。CUDA 冒烟 RTF 0.40 无回归。
- 输出 SHA256 与优化前逐比特一致(fp32 全链路); C# 测试 25/25; 导出门禁全过。

## 2026-07-02 第二轮: session 并行加载 + GQA 加速 LM 图

### ④ session 并行加载(比特级不变)
- VoxCpmPipeline 构造器中 lm/cfm/fe/vae 四组件 Task 并行创建(瓶颈是反序列化+图优化
  的单线程段, 非磁盘)。加载 4.7s → **2.25s**(CPU), 3.0s → 2.1s(CUDA)。SHA 不变。

### ③ 步耗时增长归因(先测后做)
- python 微基准(io-binding 镜像 C# 喂法): 恒定 P 实验拆分热漂移 vs 结构性 ——
  P=100→550 结构性 **+5.4ms/步**, 热漂移可忽略(±0.3ms)。
- per-node profile 归因: Transpose(注意力布局物化, ~2→4.4ms)+ Concat(present
  全量拷贝, 1.8→2.6ms)+ Split(堆叠 past 拆层拷贝)+ FusedMatMul。**MatMul 本体不随
  P 增长** —— 增长全是围绕注意力的数据搬运, 正是 GQA 融合 kernel 消除的部分。
- GQA contrib op 探针: CPU fp32 kernel 可用; do_rotary=1(非交错, 半表 cache)语义
  与手写 attention 140-150dB 吻合; **share-buffer 模式语义不符(数值错)且曾 AV 崩溃,
  弃用** —— plain 模式(present 由 ORT 分配, 回喂为 past)单层 P=550 仅 0.066ms,
  已拿走 ~90% 收益。

### ③ lm_gqa.onnx(GQA 融合注意力变体, opt-in)
- _modules.LmGqaModule/_GqaAttention: eager 前向=参考数学(与标准模块**逐位一致**,
  inf dB), symbolic 落 com.microsoft::GroupQueryAttention; KV 逐层 IO(免 Split/Concat)。
  接口: attention_mask/position_ids → seqlens_k[1](int32)+total_sequence_length(标量)。
- 打包: 与 lm.onnx **内容哈希去重**共享权重(常量折叠后权重名是 onnx::MatMul_ 流水号,
  按名字对不上; 按 dtype+dims+SHA256 匹配) → lm.onnx 转外部数据 lm.onnx.data,
  lm_gqa.onnx 仅 1.3MB。fp32/int4 都出; fp16 保标准图。ORT 门禁 117-136dB。
- C#: ILmDecoder 抽象 + LmGqaDecoder; CPU EP 且包内存在 lm_gqa.onnx 时默认使用
  (数学等价但轨迹可能分岔)。CUDA GQA 无 fp32, 保持标准图。
- 结果: fp32 短句 lm 34.5→32.5ms/步(RTF 0.955→0.932); **长句(494 步)lm 末段
  41.8→36.0ms, RTF 0.998→0.959**; int4 RTF 0.707→0.679。默认路径 SHA(fp32/int4)
  与提交基线逐比特一致; CUDA(外部化 lm.onnx)RTF 0.46 无回归; C# 测试 27/27
  (新增 2 项 GQA 功能测试)。

## 2026-07-02 第三轮: int4 包 cfm/fe int8 量化 (仅 int4 谱系, fp32 包不动)

### 关键坑: MatMulNBits bits=8 默认走慢路径
- bits=8 + accuracy_level 默认(fp32 反量化)在 CPU 上**慢 ~10x**(cfm 37→376ms/Run,
  fe 4.3→43ms)—— MLAS 只为 bits=4(任意精度级)和 bits=8+**accuracy_level=4**
  (激活 int8 计算)提供优化 kernel。
- 量化方案横评(cfm, 10 步复合 SNR vs fp32): n8 默认 376ms/34.6dB;
  **n8+acc4 17.3ms/27.9dB(采用)**; n4 12.7-25.6ms/9dB(扩散不可用);
  动态量化 MatMulInteger 9.5ms/5-7dB(激活 per-tensor 量化被 outlier 毁掉, 不可用)。
- fe: n8+acc4 1.16ms(fp32 4.25ms), SNR 32.8dB。lm 的 int4 保持默认精度级不动。

### 落地
- postprocess_dtype 增加 "int8"(weight-only MatMulNBits bits=8, acc4);
  export_cfm/run_feat_encoder 在 DTYPE=int4 时启用 + 量化门禁(cfm min_snr=22 实测
  26.2dB cos 0.9990; fe min_snr=25 实测 33.1dB)。package.py graph_dtypes 更新。
- int4 包: cfm 258→87MB, fe 244→71MB; 包总体积 ~1410→~770MB。
- 结果(9950X 8 线程): cfm 37→20.5ms/步, fe 5.4→2.6ms/步, AR 55→**35.3ms/步**;
  RTF **0.707→0.446(标准)/0.420(GQA)**, 首包 ~78ms。VAE(28.6ms 并行)未成瓶颈。
- parity smoke(en/zh/clone/stream)全过; C# 测试 27/27; fp32 包与库代码零改动
  (fp32 SHA 链不受影响)。轨迹随量化变化(同文本时长 21.0→14.2s), 音质由听感验收:
  .perf/listen_{fp32,int4_old,int4_new}.wav(旧 int4 包临时还原在 out/pkg-int4-old)。

## 2026-07-02 第四轮: int8 谱系取代 int4

### 动机: lm 的 int4 质量证伪
- 对照实验(同 fp32 prefill KV + 同输入, 单步 decode):

  | lm 变体 | 体积 | @P=40 | @P=550 | dit SNR | stop SNR |
  |---|---|---|---|---|---|
  | fp32 | 1811M | 32.1ms | 36.0ms | — | — |
  | int4 | 303M | 8.5ms | 12.0ms | 18.7dB | 36.2dB |
  | int8-acc4 | 534M | 12.5ms | 16.0ms | 31.3dB | 63.4dB |

- int4 的 stop 判停头受损是"长度跑飞"的根因(同文本 fp32 11.8s / int4 14-21s /
  int8 12.6s)。int8 每步只慢 ~4ms, 质量大幅回升 → **int4 谱系删除, int8 取代**。

### 落地
- `--dtype` 选项: fp32/fp16/int8(int4 移除); int8 = lm/cfm/fe 全部 MatMulNBits
  bits=8 accuracy_level=4 + lm_gqa 变体(量化后内容哈希去重, 707 tensors 532MB 共享,
  lm_gqa.onnx 1.3MB + 失配保险); vae 恒 fp32。INT4_BLOCK_SIZE → QUANT_BLOCK_SIZE。
- 量化门禁: lm dit prefill/decode ≥22dB(实测 25-26dB)、stop ≥30dB(35.7dB)、
  cfm ≥22dB(26.2dB)、fe ≥25dB(33.1dB); parity smoke 全过(24 项)。
- demo 默认包 → voxcpm-0.5b-onnx-int8。
- 结果(9950X 8 线程): **RTF 0.479(标准)/ 0.451(GQA)**, 首包 ~72ms, AR ~36ms/步,
  包体积 1.3GB(lm 534 + cfm 87 + fe 71 + vae 300 + embed 300)。CUDA 冒烟 RTF 0.48
  可用(GPU 首选仍是 fp32 包)。C# 测试 27/27; 产品化包与原型逐比特一致
  (SHA 17C6FC67…)。已删除: int4 包、pkg-int4-old/pkg-int8-test 临时包、
  backup-pre-b5/backup-pre-int8(旧图可由 git 历史重导出)。
