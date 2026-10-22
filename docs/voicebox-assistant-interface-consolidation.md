# Assistant 接口业务关系与合并依据

本轮按业务职责、生产实现者、实际调用方和数据访问权限复核 `VoiceBox.Assistant` 当前的 126 个接口。进一步合并三组契约，删除 5 个接口后保留 121 个；公开接口不变，每个接口仍使用独立同名文件。此前已完成五组合并，将接口数从 133 个降至 126 个，记录保留在下方。

合并依据是同一业务对象、相同生命周期及真实依赖关系，不能仅凭方法少或实现类相同就合并。阶段不同也不必然需要不同接口：如果生产代码始终以一个业务对象提供并消费这些阶段，拆出的接口没有独立使用价值。

## 业务关系

| 业务链路 | 接口的作用 | 主要协作者 |
|---|---|---|
| 音频输入与输出 | 订阅采集帧、定位渲染时间线、写入播放设备 | AudioFrontend、AudioFrameRouter、CaptureConversationPipeline、RenderReferencePipeline |
| 打断检测与确认 | 采集近端音频、观察渲染帧、积累候选、提交确认、停止已归属的采集 | ActiveNearEndCoordinator、SpeechBoundaryCoordinator、BargeInTurnCoordinator |
| 识别 | 准备 ASR 会话、输入音频、接收结果、管理回合归属与废弃会话 | RecognitionModule、RecognitionTurnOwner、RecognitionTurnCoordinator |
| 回复与工具 | 生成回复、发现和审批工具、执行工具、记录结果与模型消息 | ResponseModule、ToolUseOrchestrator、ConversationHistoryStore |
| 语音播放 | 合成语音、排队播放、接收播放事件、按回合取消和打断 | TtsPlaybackPump、SpeechPlaybackCoordinator、PlaybackInterruptionController |
| 唤醒与声纹 | 授权唤醒、验证候选身份、定位声纹证据区间 | WakeupModule、VadSpeakerGateCoordinator、SpeakerModule |
| 运行控制 | 管理运行存活状态、提交控制信号、执行 Kernel 效果、发布事件 | RuntimeRunScope、AssistantControlLoop、EffectHandlers、OrderedPublicationBoundary |

## 本轮合并的契约

| 原接口 | 合并后的接口 | 实际业务关系与改动 |
|---|---|---|
| `IAssistantToolCatalog`、`IAssistantToolApproval`、`IAssistantToolExecutor`、`IAssistantToolRuntime` | `IAssistantToolRuntime` | 前三个接口在生产代码中仅被 Runtime 继承，没有独立实现或消费者；`McpToolRuntime` 提供整组能力，`ToolUseOrchestrator` 始终通过同一个 Runtime 完成发现、授权、执行和启停。直接把三个方法并入 Runtime，删除无实际隔离作用的继承层。授权与执行仍为不同方法，只有 Allowed 才能执行；公开的 `IAssistantToolApprovalPolicy` 保持独立。 |
| `IRecognitionSessionAdmissionView`、`IRecognitionAudioPort` | `IRecognitionAudioPort` | 同由 `RecognitionModule` 实现，AdmissionView 的唯一生产消费者 `SpeechBoundaryIngress` 同时注入 AudioPort，组合根将 `recognition` 传入两次。会话查询决定当前音频能否直送 ASR，属于同一音频准入职责。合并后只注入一次，查询和投递使用同一对象；会话身份查询、采集已挂接检查各自的原有语义保持不变。 |
| `INearEndDiagnosticView`、`INearEndConfirmationStore` | `INearEndConfirmationStore` | DiagnosticView 的唯一生产消费者 `ActiveNearEndControlHandler` 已持有 ConfirmationStore。两个端口始终来自同一个 `ActiveNearEndCoordinator` 或 `NullActiveNearEnd`，用于确认／丢弃打断后报告该对象的状态。将快照方法并入确认存储，删除重复字段、构造参数和装配转换，确保诊断与确认对应同一对象。音频回调仍通过各自的采集、渲染和端点接口工作。 |

## 此前已合并的契约

| 原接口 | 合并后的接口 | 依据与代码变化 |
|---|---|---|
| `IActiveNearEndRuntimeView`、`IRuntimeLivenessView` | `IRuntimeLivenessView` | 都只有 `IsStopping`，生产实现都是 `RuntimeRunScope`。近端、采集和渲染流程统一读取同一个运行存活事实。 |
| `INearEndSpeakerTimelinePort`、`ISpeechTimelineLocator` | `ISpeechTimelineLocator` | 前者只是没有新增成员的派生接口，生产调用方已经使用后者。近端与普通语音的时间线定位统一使用共享契约。 |
| `ISpeakerGateResetPort`、`ISpeakerGateCapturePort` | `ISpeakerGateCapturePort` | Reset 接口没有独立调用方，只作为 Capture 接口的父接口存在。将 `Reset` 直接并入 Capture，保留 Boundary 与 Capture 的现有继承关系和权限。 |
| `IToolActivitySink`、`IToolTranscriptSink`、`IConversationToolTranscriptPort`；`IToolActivitySource`、`IToolTranscriptSource` | `IToolHistorySink`、`IToolHistorySource` | 唯一生产记录者都是 `ConversationHistoryStore`，唯一生产发送者都是 `ToolUseOrchestrator`，组合根始终绑定同一个历史实例。其中两份消息记录契约的签名完全相同。合并为一次绑定、一个字段，确保执行审计与工具消息属于同一历史接收者；两种记录的原有时机保持不变。 |
| `IRecognitionPreparePort`、`IDeferredRecognitionReadyPort` | `IRecognitionPreparationPort` | 唯一生产实现都是 `RecognitionTurnCoordinator`，唯一外部消费者都是 `BargeInTurnCoordinator`，原来通过 `BindRecognition(recognitionTurns, recognitionTurns)` 成对绑定。会话准备与会话就绪前的延迟标记共同服务于识别准备流程，现在通过一个参数绑定。 |

## 继续保留的业务边界

| 接口组 | 保留原因 |
|---|---|
| `INearEndCapturePort`、`INearEndRenderPort`、`INearEndEndpointPort`、`INearEndConfirmationStore`、`INearEndCaptureControlPort`、`INearEndResetPort` | 虽有相同实现者，但调用方分别是实时采集、双工渲染、VAD 入口、控制确认、语义打断和生命周期。整体合并会让音频回调获得确认或重置能力，也会让控制消费者接触 PCM。确认处理器使用的诊断快照已并入确认存储。 |
| `IRenderReferenceSource`、`IDuplexAudioFrameSource` | Render 接口虽没有新增成员，却标记帧是否具备可用于双讲检测的渲染参考。这是能力组合需要区分的业务事实。 |
| `IRuntimeLivenessView`、`IRuntimeCancellationView` | 前者只读取停止状态，后者提供异步工作的取消令牌；实时观察与异步任务取消有不同消费者。 |
| `IAssistantTool`、`IAssistantToolApprovalPolicy`、`IAssistantToolRuntime` | 分别是公开工具扩展点、公开的宿主授权策略和内部工具会话。前两者由宿主独立实现、注册，Runtime 组合这些能力，因此保持各自的接口。 |
| `IConversationPromptPort`、`IConversationRollbackPort`、`IConversationBackgroundTextPort` | 虽由同一历史存储实现，但分别服务于回复建模、语义打断回滚和后台结果入库，写入权限与触发条件不同。 |
| `IRecognitionTurnLifecyclePort`、`IRecognitionBoundaryPort`、`IRecognitionSessionCleanupPort` | 回合准备、语音边界观察和废弃 provider 会话清理有不同所有者与消费者；不能把回合状态重置与 ASR 资源清理混为一个操作面。 |
| `IResponseGenerator`、`IResponseGeneratorLifecycle` | 文本生成器只需生成流；工具生成器还拥有后台工作和运行生命周期。生命周期仍是可选能力，避免要求普通生成器实现空操作。 |
| `IBackgroundToolResultSource`、`IToolHistorySource` | 后台结果交给播放与发布流程，工具历史交给历史存储；接收者、数据用途和生命周期不同。 |
| `IPlaybackStartedSink`、`IScheduledSpeechSink`、`IPlaybackTextEchoResetPort` | 分别由播放控制边界和文本回声保护实现；播放开始、文本调度和回合重置的触发点不同。 |
| `IPlaybackInterruptionPort`、`IPlaybackCancellationPort`、`IActiveNearEndPlaybackPort` | 分别承担语义打断、播放所有者取消、近端临时压低音量，权限和消费者不同。 |
| `ICandidatePreRollSource`、`IValidationCapturePreRollPort` | 候选预录音读取与验证阶段积压音频的所有权移交不同，后者需要取走缓冲区。 |
| 各类 Factory、EffectHandler、ControlSignalPort 和诊断观察接口 | Factory 负责实例创建，Handler 负责副作用，SignalPort 负责控制入口，诊断接口只观察事实。保留各自作用，避免跨职责聚合。 |

## 验证

更新实现者、组合根、测试替身及架构断言，并同步架构文档。原有工具历史回归继续覆盖前台与后台的执行审计和消息记录；本轮增加工具授权的 Allowed、Denied、RequiresConfirmation 三种回归，验证合并后不会跳过授权。识别音频回归通过合并后的接口检查会话准入、回合关联、音频投递以及结束后清理。近端确认回归通过合并后的接口读取确认前后的诊断状态。

`dotnet test src/Tests/VoiceBox.Assistant.Tests/VoiceBox.Assistant.Tests.csproj --no-restore --verbosity quiet`：1,049 项全部通过，包含 MCP 集成、架构约束及运行时回归。
