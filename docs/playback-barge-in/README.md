# VoiceBox 播放期打断修复方案

实施目标是让助手在用户插话获得授权时及时停止当前 TTS，并把这次插话从句首到句尾交给同一个 ASR 会话。本文适用于助手处于 `Speaking`、TTS 正在播放且允许打断的场景。已有实验基于提交 `96aead1`，实施时以当前检出的代码为准。

## 1. 停播判定

| 说话人确认 | 事件先后 | 期望行为 |
|---|---|---|
| 开启 | DTD 先发现近端语音，甚至已达到 `Interrupt` | 查询本次候选的声纹结果。已有本人 `Match` 就立即请求停播。`Pending` 期间按原增益播放，匹配结果就绪时立即处理。 |
| 开启 | 本次候选先获得本人 `Match`，DTD 仍为 `Continue` | 立即请求停播。已定位到采集时间线的 VAD 语音也可建立候选。声纹结果到达后，即使没有下一帧 DTD，也要完成停播。 |
| 开启 | `Mismatch`、证据不足、超时或候选已取消 | 当前回复继续播放。后续插话可建立自己的候选。此前唤醒或上次插话的匹配结果不能授权本次打断。 |
| 关闭 | DTD 首次判定 `Interrupt` | 立即请求停播，声纹查询次数为零。此前的弱证据可沿用已有临时降音量策略。 |

**关于“此前的匹配”**：当前 verifier 的 `RecentOwnerPrior` 和 `AllowStrongUpstreamOneVote` 只在与当前候选的近期匹配窗同时成立时降低票数，仍需当前窗 `Match`，不会单独复用旧判决授权。若验收要求“上次匹配零影响”，需为该用途关闭这两个先验，而不是假设代码已经如此。

**“立即”的验收方式**是从授权成立开始暂停后续 DTD 回调，并暂缓主识别音频交付。让控制队列处理已收到的事件，当前回复应已执行一次物理停播。随后恢复音频交付，ASR 应能收到整句。设备已提交的尾音另行测量。

## 2. 已复现的问题

现场日志在 `22:15:06` 记录了候选 3 的 `HighConfidenceMatch`，得分 `0.6264`，阈值 `0.3000`，同时 `dtdInterrupt=False`。随后 DTD 已两次达到 `260/180 ms`、`220/180 ms`。候选一直是 `True/True`，`activeTurn=0`，直到 `22:15:14` 播放结束后被清除。

在故障版本中，声纹模型在后台工作线程推理，播放期候选由双工回调查询结果。授权后，候选还会被新的 DTD 帧覆盖 `Decision`。取识别音频的两条分支 `main-vad-timeline` 和 `main-candidate` 又都要求主链到达这个终点。现场八个快照中，主链始终落后 360–680 个采样点（16 kHz 下为 22.5–42.5 ms）。两条链都在前进，追赶目标也一直在移动。[候选状态机](../../src/Assistant/VoiceBox.Assistant/Modules/BargeIn/State/NearEndCandidateStateMachine.cs) · [打断数据流](../../src/Assistant/VoiceBox.Assistant/Modules/BargeIn/Controllers/NearEndDataPlane.cs)

隔离实验交叉组合了声纹开关、有无已定位 VAD 句首、主链落后 360 或 680 个样本。原实现开启声纹的 4 组均失败，关闭声纹的 4 组通过。固定授权时的 `Decision` 后，8 组均在主链补齐目标的第一个回调提交，拼接后的 PCM 与期望区间逐样本一致，连同已有相关用例共 282 项通过。**这个实验验证了窗口追赶故障。第 1 节要求的“停止交付新帧仍能停播”需要独立实现和测试。**

## 3. 实施路径

1. **让声纹结果能主动唤醒打断流程。** 在现有 DTD 或已定位 VAD 的入口建立候选，绑定 `BargeInCommand`、`candidateId`、`ownerTurnId`、`streamEpoch` 和候选代次。缓存立即命中与后台推理稍后命中都进入同一授权处理。后台先订阅声纹缓存的更新通知，再读取本候选的最新采样范围和实际推理窗口。新窗口独立计时，同一个仍在运行的窗口保留原截止时间。每个候选只保留一个有效等待，换轮、取消和关闭时释放。音频回调保持非阻塞。[声纹结果接口](../../src/Assistant/VoiceBox.Assistant/Modules/Speaker/Contracts/ISpeakerEvidenceProvider.cs) · [验证器](../../src/Assistant/VoiceBox.Assistant/Modules/Speaker/Controllers/SpeechGatedSpeakerVerifier.cs)
2. **让控制面收到授权即可停播。** 在授权时固定候选的判定窗口和已知句首，保留后续读取该段 PCM 所需的时间线信息。先登记这次插话的采集归属，再由控制面校验回复、候选和代次仍有效，复用 Kernel 的单次停播入口。`TryStartCapture` 与预滚快照从停播的前置条件中移出；ASR 会话就绪和转写结果本来就不在停播路径上（ASR 音频端口在会话建立前会缓冲），新设计也不得把它们引入。生产端只投递事件，设备清队列由既有播放控制路径执行。重复通知、迟到的匹配和同时到达的 DTD 事件均只能作用于对应候选一次。[控制面入口](../../src/Assistant/VoiceBox.Assistant/Modules/BargeIn/Controllers/ActiveNearEndControlHandler.cs) · [打断回合协调](../../src/Assistant/VoiceBox.Assistant/Modules/BargeIn/Controllers/BargeInTurnCoordinator.cs) · [Kernel 停播入口](../../src/Assistant/VoiceBox.Assistant/Kernel/TurnKernel.cs)
3. **让采集跨过停播时刻。** 停播会立即使 `playback=False`，控制面也可能离开 `Speaking`。将已授权、仍待补音频的候选交给新打断回合持有。检查 `cleared:playback-inactive`、回合切换及清缓存路径，避免这次停播清掉自己的预滚或句尾。确认之后播放继续关闭，即便最后识别为空或过滤为回声。保留现有控制队列入队失败、取消和采集超时的收尾路径，确保下一次对话可用。[确认事务](../../src/Assistant/VoiceBox.Assistant/Modules/BargeIn/State/NearEndConfirmationJournal.cs) · [设备停播](../../src/Assistant/VoiceBox.Assistant/Modules/Playback/Controllers/PlaybackInterruptionController.cs)

**采集与停播的衔接。** 授权时冻结 `AuthorizedDecision`，由 Kernel 对当前回复执行一次停播。若 VAD 已报告语音开始、首段 PCM 尚未定位，打断回合先持有采集时间线。收到首段后按实际起点补预滚。早于本次语音证据的旧句尾会被丢弃。待采集回合登记一次性定时器，采集激活或取消时释放。设备停止回调时仍能按预算收尾。

## 4. ASR 的句首和句尾

- **句首**取自同一 `streamEpoch` 的处理后麦克风音频。优先取本次已定位的 VAD 起点并补配置的 pre-roll。缺少可靠 VAD 时，按当前模式使用候选起点或已保留的判定前历史。声纹比对窗可能只覆盖句中，`Match` 的完成时间也可能晚于整句话。若 VAD 先报句尾而声纹稍后才匹配，仍要保留这句短命令。
- **中段**以已交付区间的排他终点（下一个待交付的采样位置）为水位。预滚、等待控制面期间的缓存和后续实时 PCM 按半开区间 `[start, end)` 连续交接。ASR 会话较晚建立时先补预滚，再补等待期间的音频，每个样本只交付一次。
- **句尾**结合当前语句的 VAD 结束事件或连续静音确定。停播后麦克风继续采集，等待主链补齐有效终点、在途回调和 ASR 队列排空后再完成识别。在可恢复的收尾期间重新出现本句语音，应继续收音。对真正的流重启、断帧和排空超时记录实际缺样。旧 epoch 的坐标不能拼入新流。[采集与 VAD 回调顺序](../../src/Assistant/VoiceBox.Assistant/Runtime/Pipelines/CaptureConversationPipeline.cs) · [采集、去重和排空状态机](../../src/Assistant/VoiceBox.Assistant/Modules/BargeIn/State/NearEndCaptureStateMachine.cs)

## 5. 验收

| 用例 | 如何控制输入 | 必须观察到的结果 |
|---|---|---|
| 本人匹配，两种顺序 | 一组先给 `Interrupt`、阻塞声纹再返回 `Match`。另一组用 VAD 建候选、DTD 保持 `Continue`。匹配前暂缓主链，匹配后不再推 DTD 帧 | 匹配前停播 0 次，匹配后仅运行控制队列即物理停播 1 次。恢复主链后得到完整 PCM。覆盖同步命中与异步命中。 |
| 关闭声纹 | 注入弱近端证据、纯播放回声，再单独注入可确认的 `Interrupt`。暂缓主链 | 弱证据和纯回声均不物理停播。`Interrupt` 后停播 1 次，声纹调用 0 次。 |
| 非本人、陈旧结果和竞态 | 注入拒绝、取消、超时、换轮后的旧 `Match`、重复 `Match`，并交错主链补帧与授权通知 | 相应拒绝和旧结果停播 0 次。合法候选恰好停当前回复 1 次，只建一个识别回合。 |
| 滑动窗口和音频边界 | 重建第 2 节的 8 个组合，让两条链持续前进。再覆盖句尾先于匹配、延迟连接 ASR、句中停顿恢复、停播后仍在说话 | 生产入口触发停播。用带位置标记的 PCM 比较完整区间、长度和顺序，首尾齐全，无重复、无跨轮混入。 |
| 真人中文回放 | 声纹开启时分别使用登记本人和非本人录音。关闭时复用相同语音。包含短句、弱句首、长句停顿、弱尾音和纯 TTS。正常连续采集与模拟流重启各自报告 | 每例同时记录预期停播、实际设备停播时间、语音参考区间、实际送入 ASR 的区间、参考文本、识别文本及字符错误率（CER）。短句查首尾字，长句查停顿后的内容。 |

**已知中文反例。** 可从现有数据集复跑 `e2e-002-aishell1-test-BAC009S0769W0161`，参考文本为“继前年和去年连续两年突破四万亿元后”。关闭声纹时，默认是连续采集；以 benchmark 的 `--stream-restart` 压力模型（模拟 AEC/采集时间线重启）复跑时，回放在 280 ms 发出停播，但 ASR 在 2.68 秒已收尾，参考语音延续至 6.76 秒，转写仅为“以前”。日志中的 `aligned` 和 `missing=0` 仅说明送达了所选终点。按 Demo 的连续采集行为及 VAD 阈值重放，ASR 收到 `[1.20 s, 7.17 s)`，覆盖参考区间 `[1.94 s, 6.76 s)`，转写为“与前年和去年连续两年突破四万亿元后”。该样本可同时检查过早断句和首字识别。这一组早期对照覆盖关闭声纹。本次双模式复测结果见下表。E2E 构建链默认关闭本人验证；传入 `--speaker-model-dir` 与 `--speaker-profile` 时会实际加载 CAM++ 服务和注册资料，验收开启模式时应使用这两个参数。[评测入口及运行说明](../../src/Assistant/Test/VoiceBox.Assistant.AecChallenge.Benchmark/README.md) · [E2E 构建与采集重置配置](../../src/Assistant/Test/VoiceBox.Assistant.AecChallenge.Benchmark/EndToEndEvaluator.cs)

在仓库根目录运行相关回归，并把新增用例纳入过滤范围。相关测试入口包括 `VoiceAssistantBargeInVadCandidateTests.cs`、`VoiceAssistantBargeInInferenceDeadlineTests.cs`、`VoiceAssistantBargeInEndpointRegressionTests.cs`、`VoiceAssistantLifecycleTests.NearEndDrain.cs` 和本方案新增的 `PlaybackBargeInDecouplingTests.cs`（移动判定窗、待采集、无后续音频帧停播、采集超时、Kernel 拒绝回滚）。C# 修改遵循根目录 [AGENTS.md](../../AGENTS.md)。

```powershell
dotnet test src/Assistant/Test/VoiceBox.Assistant.Tests/VoiceBox.Assistant.Tests.csproj --filter 'FullyQualifiedName~BargeIn|FullyQualifiedName~NearEnd|FullyQualifiedName~RenderAwareDtd'
```

### 2026-11-21 复测记录

修复前的隔离用例复现了两种跨窗声纹超时、旧句尾误结束、首个 VAD 音频块迟到导致丢句首，以及采集时钟停摆后不收尾。实际运行旧版 CAM++ 评测还触发了 995 ms 窗超过 800 ms 实时上限的配置错误。修复后，回归直接比较送入 ASR 的 PCM，并在停止音频回调后检查授权和收尾。声纹 worker 的 Debug 日志恢复了每个候选任务的排队和推理用时。Assistant 全量测试 1097 项通过，打断相关 331 项连续三次通过，评测器自检通过。

| 声纹与回放条件 | 设备停播与转写 |
|---|---|
| 关闭，标准双讲样本 `e2e-002`，连续采集 | 280 ms 停播，转写遗漏首字“继”。 |
| 关闭，同一样本，启用 `--stream-restart` | 首次停播 280 ms，设备队列在 290 ms 排空，本次转写为“四万元”。旧评测会让后续清队列覆盖首次停播时间，因此此前记录的约 3.6 秒应更正。 |
| 开启，独立注册后重放原始场景 | S0769 本人样本 `e2e-002` 未触发，跟踪记录显示声纹多数窗判为不匹配。S0767 非本人样本 `e2e-001` 未触发。 |
| 开启，使用同一注册资料，在干扰较轻的对照场景重放 | 去掉附加噪声及近端房间响应，信号回声比设为 12 dB。本人在 740 ms 由 `speaker-match` 停播，转写为“即前年和去年连续两年突破四万亿元后。”，非本人未触发。 |

独立注册用了 S0769 的另外 8 段录音和 S0764 的背景声。早先单录音建档的探索实验中，本人曾在 480 ms 停播，也有非本人误接收。那条非本人录音曾参与生成注册背景声。以上是单条样本观测。声纹误拒、误接收、首字转写及重启后的整句完整性仍需在更大的独立样本集上评估。

实施报告需分别给出声纹开关两组结果。用同一候选标识串起“插话开始 → 授权就绪 → 控制面接收 → 设备执行停播”，并另记实际采集起点、终点、缺样数和转写文本。以事件顺序验证软件响应，以真实回放测量实际延迟。现场的 180 ms 声学门槛和 637 ms 单次声纹耗时均不作为两种模式通用的延迟指标。
