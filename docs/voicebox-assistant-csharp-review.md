# VoiceBox.Assistant C# 审查记录

日期：2026-10-18。依据：`csharp-guidelines`。基线提交：`fde0d45`。

对项目的 471 个非生成 C# 文件做了静态扫描，重点检查了公共入口、配置验证、异步任务监督、事件发布、对话历史、分句和音频时间线。静态扫描不等于对全部文件逐行审计；以下结论来自重点代码检查、回归测试和局部性能测量。

## 已完成的修改

| 位置 | 问题与修改 | 对应指南 |
| --- | --- | --- |
| `Modules/Response/Formatting/SentenceSplitter.cs` | `Push` 使用迭代器延迟修改缓冲区；未枚举时 `Flush` 看不到新文本，重复枚举会再次追加文本。改为调用时处理并返回已物化的 `IReadOnlyList<string>`。固定标点集合改用字符模式匹配。 | AV0112、AV1130、AV1250、AV1705、AV2301 |
| `Modules/Speaker/Buffering/SpeakerAudioTimelineBuffer.cs` | 语音时长统计逐样本取模，且帧终点相加在 `long.MaxValue` 附近会溢出。改为每帧计算一次环形偏移，拆成最多两个连续片段扫描，并按实际帧长推进。保留有限样本筛选和能量累加顺序。 | AV1115、AV1500、AV1532 |
| `Api/Configuration/SpeakerConfirmationOptions.cs` | 两处说明中的默认值与初始化值不符。推理超时文档改为 150ms，近期身份先验文档改为 5000ms。 | AV2301、AV2306 |
| `VoiceAssistantGoldenTraceTests.cs` | 场景测试只等待 `Thinking` 状态，就触发下一次打断；状态已变更并不保证思考文本已经发布，导致超时或轨迹缺失。改为等待首个思考文本事件，再触发打断。完整轨迹断言保持有效。 | AV1605、AV1618 |

新增 6 个回归测试，覆盖分句器的枚举时机、重复枚举、跨片段累计，以及音频统计的环绕、尾帧、非有限样本和时间线终点。

## 验证结果

- 修改前：1048 项通过，`GoldenTrace_BargeInBeforeResponseGenerated_IsExact` 超时；单独复跑又出现轨迹不一致。
- 修改后：Release 全量测试 1055 项全部通过；使用最高编译器警告级别并将警告视为错误。
- 修正后的场景测试连续独立运行 10 次，全部通过。
- `git diff --check` 通过。

全量验证命令：

```powershell
dotnet test src\Tests\VoiceBox.Assistant.Tests\VoiceBox.Assistant.Tests.csproj --no-restore --configuration Release -warnaserror -p:WarningLevel=9999
```

## 音频统计微基准

使用独立的 .NET 10 Release 控制台，将基线提交的缓冲区实现与当前实现放入同一进程比较。环形容量为 96000，起点为 90037，包含跨环绕窗口和非有限样本。随机种子为 17；200 组不同起点、长度和阈值的查询结果一致。

每种窗口先交替预热 2000 次，再进行 5 轮测量；每轮每种实现查询 4000 次，轮流交换测量顺序，取中位数。

| 窗口样本数 | 基线每次查询 | 优化后每次查询 | 加速比 | 优化后查询分配量 |
| --- | ---: | ---: | ---: | ---: |
| 8000 | 21.20µs | 3.45µs | 6.14× | 4000 次合计 0 字节 |
| 48000 | 130.63µs | 20.95µs | 6.24× | 4000 次合计 0 字节 |

这些结果描述本机语音时长统计方法的性能，不代表整个助手响应延迟的改善幅度。

## 结构性审查意见

`ActiveNearEndCoordinator` 同时协调候选身份、音频尾部补齐和播放音量恢复，文件约 2079 行；`RecognitionModule` 约 1091 行；配置验证入口也包含大量连续校验。这些是 AV1000、AV1500 下值得继续拆解的维护热点。拆解时应沿现有类型化端口与状态所有权边界推进，并保留当前锁和事件轨迹语义。

项目的架构测试明确约束 Runtime/Modules/Composition 中的委托依赖。后续抽取公共逻辑时，应先检查这些边界，避免为了减少重复代码引入新的回调依赖。
