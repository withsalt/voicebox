# Render-Aware DTD 训练轮次台账

方案：[render-aware-dtd-training-plan.md](render-aware-dtd-training-plan.md) · 基线：[render-aware-dtd-baseline-20261012.md](render-aware-dtd-baseline-20261012.md) · **归档**：[render-aware-dtd-archive-20261019.md](render-aware-dtd-archive-20261019.md) · **决策包**：[render-aware-dtd-decision-pack-20261019.md](render-aware-dtd-decision-pack-20261019.md)

每个轮次一行汇总。判定列取值：`达标候选` / `回退` / `中性`。数据集列为"保留（成为最佳）"或"已删（配方保留于 round.json）"。

> **约束更新（2026-10-16，用户）**：**禁止多模型融合方案**（union/veto/多链——推理成本过高）；**达标候选必须是单模型推理**。历史融合链（R4–R13）作为诊断证据留存但不再构成达标载体（R11 链 7/11 因此不视为达标候选）；单模型前沿（截至 R13）：无单模型同时通过 B1+C1+B2+B3+early。

| 轮次 | 日期 | 假设/唯一变量 | 数据集 | 训练 | 选点/导出 | acoustic | e2e | compare | 判定 | 数据集 | 下一步 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 2026-10-12 | 基建（工具重建+排除清单+链路自证） | — | — | e147 复现：唯一合格点 | 全量重放与基线逐字段一致 | 决策级与基线一致；CER/切片延迟非确定 | acoustic Passed；full 噪声型 Regressed | — | — | 启动 R1 |
| R1 | 2026-10-12 | 设备族覆盖扩展（real 池+7 语料，纯数据层） | v8r1（13,759→26,138 行；验证集=v5 逐位；逐位对拍 PASS） | ✅230/230 完成 | e224/e213（anchor 严格非回退=0；验证集被稀释） | e224: 召 0.9328↓、FP/h 7.864/10.514（B1 微过）、P95 1973 持平；**目标族 FP 首次下降**（icassp 18→13、mobile 6→3）但 noisy 7→14 | 未跑（P7 veto） | 官方 acoustic-only：Regressed 207 findings（新 FP 15/新 FN 11） | **否决** | 保留（R2 复用） | R2=曝光配比修正 |
| R2 | 2026-10-13 | real 桶权重 0.20→0.40（恢复旧族曝光） | 复用 v8r1（不动） | ✅230/230 | e208（验证集工作点首次与 anchor 等高：fpU95/P50 全同、r@0.1% 反超） | **FP 33→21（-37%）**、FP/h 4.95/7.12；但召回崩塌（0.8742↓、FN 27→58）、P95 2144↑、1 新早触发 | 未跑（P7 veto） | 官方 acoustic-only：Regressed 290 findings（新 FN 33/新 FP 7/新早触发 1） | **否决** | 保留（R3 复用） | R3=工作点重选（0.55→0.48） |
| R3 | 2026-10-13 | 运行阈值 0.55→0.48（验证集预登记扫描选中；模型=R2 e208，无重训） | 复用 v8r1 | —（无训练） | 扫描报告 selection/threshold-sweep.json | 召回仅 +0.65pt（0.8807）、FP 21 持平、P95 仍 2144、早触发 1 | 未跑（声学门 veto） | — | **否决**（阈值已回退 0.55） | — | 续入 R4 |
| R4 | 2026-10-13 | 二次确认/否决融合（base=e147 + gate=R2，soft-min lo=0.05，单 ONNX 同契约） | 复用 v8r1 | —（无训练） | 验证集预登记分析（veto-rule-analysis-hardgate/soft） | FP 33→24（新增 0✓）FP/h -28%；召回 -1.3pt（否决 6 TP）、P95 持平、早触发 0 | 未跑（声学门 veto） | — | **否决**（standalone）；融合工具链验证 ✓，FP:TP 交换比 1.5 | — | R5 落地后 [快基+保守门] 融合 |
| R5 | 2026-10-13 | onset 损失 0.10→0.30（单变量直指 P95 延迟；数据集=v5 与 candidate-01 全同） | 复用 v5 | ✅219/219 早停（best=188） | e165/e188/e191（e165 首次验证集三轴不劣 anchor） | 见下：standalone 均被否决（FP 38/51/60） | 全量 e2e+compare（union+veto lo0.02 链，首次） | 见 R5 详情 | **否决**（无单配置过全部门）；融合前沿大幅推进 | — | R6=onset0.30×v8r1（已启动） |
| R6 | 2026-10-13 | onset 0.30 × v8r1 数据集（组合 R5 召回机制 + R1 设备族 FP 机制） | v8r1（复用） | ✅230/230（best 213，~7.6h） | e144（首个验证集双轴严格更优）/e213（弃） | e144 standalone：**B1 双降 7.142/9.686（模型级首次）** + noisy 27/30；召回 .9371 | 全量 e2e+compare（lo=0.02 链） | **机器判定 6/9 通过（历史最佳）**：A3/B1/B3/C1/C2/C3 PASS；A1 仅 4 case 翻转；B2/B4/B5 精确持平 | **否决**（未全项达标） | 保留 | R7=TTS 渲染 far 域 |
| R7 | 2026-10-14 | far 池注入 TTS 渲染域（237 条 VoxCPM 离线合成 ×100 份额 16.5%；其余=R6） | v8r2（v8r1+farend_train TTS 行；验证集=v5 逐位镜像） | ✅230/230（best 220） | 无 anchor_nearest（姿态突变）；e220/e213 | e220 standalone：B1 双降 7.135/9.678 但召回 .9219；链被 R6 链支配 | **e2e 双层协议首次执行**（3 实验） | R7链(s=0.5) e2e 逐位≡基线（增量未开火）；**e220 standalone e2e 首次移动**（heard 74→80 + echoOnly 6→12 + FA 64）；s=1.0 链 C2 .908↑ 但 FA 74 | **否决** | 保留（v8r2/语料留作 R8/R9 底） | R8=份额→33%；R9=多音色（参数待 R8 定） |
| R8 | 2026-10-14~15 | TTS far 剂量翻倍（样本份额 11.4%→21.3%；multiplicity 100→250） | v8r3（farend_train ×250，行份额 33.0%；其余=v8r1；验证集=v5 逐位镜像） | ✅223/230 早停（best 192，valAP .9654） | e185（anchor_nearest，验证集双轴严格更优）/e192/e196 | **standalone 全崩**：e185 召回 .8134（device 族双讲塌陷，.78→.48 等）、FP/h 5.916/8.262；链 s=1.0：召回 .9328、FP/h 7.649/10.275 | **e2e 双层**：e185 heard 75（+1）/echoOnly 11/FA 59.2；s=1.0 链 heard 74≡基线/echoOnly 14/FA 74.0 | standalone **FAIL 8/11**（severe 557）；链 **FAIL 6/11**（severe 416） | **否决（剂量轴反噬确证）** | 保留（v8r3；被 R6 支配，可删） | R9=多音色@匹配剂量（mul 80，用 R8 剂量结论定参） |
| R9 | 2026-10-15 | 多音色 TTS 语料（vc237+q3-70，6 音色）@ 剂量匹配 R7（mul 80，行份额 16.96%，realized 11.64%） | v8r4（重建：0 失败；验证集 v5 镜像 3000 文件 0 差异） | ✅230/230 全训（best valAP 0.9659 ep182——历史最高） | e170（anchor_nearest）/e220/e182 | **standalone 全崩**（同 R8 型）：召回 .8373/.8590/.8959（−4.5~−10.4pt）、FP/h 5.68~7.88；链 s=1.0：recall .9328 但 FP/h **8.634/11.401 高于基线** | **e2e 双层**：e170 heard **73<74**（首次跌破）/echoOnly 12/FA 64.1；s=1.0 链 heard 73/echoOnly 14/FA 74.0 | standalone **FAIL 8/11**（severe 587；C2 首次跌破）；链 **FAIL 8/11**（B1 首次失守） | **否决（多音色假设失败；TTS 轴判耗尽）** | 保留（v8r4，可删） | R10=容量叠加（4L/128bands × R6 配方） |
| R10 | 2026-10-15 | 容量叠加：gru-layers 3→4 + frequency-bands 80→128（× R6 配方 onset 0.30 × v8r1；组合不可拆照 H 轮先例） | 复用 v8r1（验证完好：48000+1500，val 抽样 20 文件 0 diff） | ✅230/230 全训（best valAP 0.9661 ep213；**anchor_nearest 池=3** 历轮首次多点合格） | e157（rL95 .716 ≥ anchor；fpU95 47.351 恰等于 anchor）/e205/e213 | **B1/B3/P50 三线齐进**：e157 fp/h **7.383/9.964 双降**、noisy-move **28/30（rL95 .787，历史最强）**、P50 368（−28ms）；但 C1 −0.5pt、P95 2132+、每点早触发 1 | **e2e 全面恶化**：e157 heard 75（+1）但 echoOnly **18**/FA **93.7**；链 heard 75/echoOnly **20**/FA **103.6**（链族最差） | standalone **FAIL 8/11**（severe 492）；链 **FAIL 8/11**（severe 482） | **否决**（链被 R6 链双向支配） | 保留（v8r1 为 R6 遗产） | R11=真实双讲弱标签（backlog 最后未试数据轴） |
| R11 | 2026-10-15~16 | **真实双讲弱标签**：四大真实族 `_doubletalk_mic/_lpb`（~13k 会话从未入训，无近端标签）→ 会话内回波路径拟合（频域 Wiener 4096-tap，far-only ERLE 中位 +17.5dB）→ 残差能量弱标签（官方同款 `labels_for` 派生）→ 确定性替换 v8r1 mixture 槽位 **15%（3,610 样本，占全集 7.5%）** | v8r5（v8r1 hardlink 克隆 + 注入；标签复检 coverage diff p50 0.0015；验证集 v5 镜像 0/50 diffs；v8r1 原件 100/100 独立 inode） | ✅230/230 全训（best valAP 0.9645 ep174；**anchor_nearest 池=5** 历轮最多） | e181（rL95 .7196 ≥ anchor、fpU95 44.68 < 47.35 双轴优）；e212/e174 | **standalone e181 历史性突破**：recall **.9696（纪录）**、rL95 .9497、**P95 1892（B2 首次实质下降）**、early 0、noisy **29/30（.8333）**、设备族双讲全面改善（android/fullband_mobile 22/22 满分）；代价 FP 42.6 | **链 union(e147,0.8·e181)+veto(e157,lo=0.20)：声学全目标严格通过（项目首次）**——B1 7.406/9.996、B2 1937、B3 28/30(.7868)、C1 .92625、early 0；链 e2e：heard 76（+2）但 heardP95 1675/B5 54.3/C3 1677.5 | 链 **PASS 7/11（历史最佳）**：A3/B1/B2/B3/C1/C2；severe 169（case 翻转 13：10 FP + 3 FN，比对文件口径）；standalone PASS 5/11 | **否决（未全达标）但突破轮**：声学合取首次清空；剩余墙 = e2e 层 + A1 10 翻转 | 保留（v8r5 为 R12 底） | R12 = 真实双讲剂量 15%→30%（v8r6） |
| R12 | 2026-10-16 | 真实双讲剂量 15%→30%（v8r6，7,156 替换；其余=R11） | v8r6（映射与校验同 R11 流程） | ✅早停 ep195（patience 30；best valAP 0.9639 ep165） | e181（anchor_nearest，rL95 .7174/fpU95 46.02 双轴优）/e165 | standalone 双双反噬：e181 召 .9653 但 FP 91.9/P95 2583/early 3；e165 召 **.9718** 但 FP 97.2/P95 2410/early 2 | 未跑（P7 veto） | 链同参数对照（vs R11 lo020）：recall .9479、fp **7.157**（更深）但 **P95 2144 ✗、early 1 ✗**；lo030 同型 | **否决**（剂量反噬；15% 为最优） | 保留（可删） | R13 = 多增量链工具 + 融合空间探索（免训练） |
| R13 | 2026-10-16 | 融合空间探索（免训练）：双增量链工具扩展 + e147+0.6·e144+0.8·e181+veto(e157,lo0.20) | —（复用） | —（无训练） | 回归自证：R11 链重导逐位一致（0.0 diff） | 四模型链：recall .9501（与 R11 链逐位相同）、fp **7.656**（+0.25 劣）、P95 1937/early 0 保持 | 未跑 | 链 case 翻转 **14**（vs R11 链 13） | **否决**（e144 增量零贡献纯加劣；融合探索收敛于 R11 链） | — | 剩余：A1 13 翻转 + e2e B4/B5/C3；契约扩展为最后杠杆（需批准） |
| R14 | 2026-10-16~17 | **ONNX 输入契约扩展（用户批准，方向不限；v1= AEC 输出第三通道）**：mic（AEC 前）+对齐 render + **post-AEC 第三通道**；训练侧离线谱代理通道；模型前端 2→3 输入 | v8r5（R14b）→ 真 APM sidecar 混通道（R14c，否决） | R14a 作废；R14b 完成（早停 @ep205，best sel 1.37379 @ep175）；R14c 230/230（混通道，否决） | 4 输入导出+parity ✓；**通道语义契约化**（aec_channel_source metadata） | e181：capture 模式 FP **480** → **proxy 运行时模式 FP 64（15.71/h）、recall .9566/L95 .9339、P50 376、P95 2037、early 1**（通道一致后灾难消除；仍被基线支配） | 未跑 | — | **否决**（R14b/c/e 均未过 B1/B2）；但机制打通+通道一致性铁律确立 | 保留（v8r5 为 R11 遗产） | **R15 = 基线家系 v5 + v6 冻结配方 + 同一第三通道**（已启动 2026-10-17 02:48） |

### R15–R34 汇总（单模型时代，2026-10-17 ~ 10-19）

| 轮次 | 假设/唯一变量 | 数据/训练 | 评定点 | acoustic（基线 .9414/.9161、33/7.903、396/1973、early0、B3 .7032） | e2e（基线 heard .8506/P95 1475、FA 29.60、drain 1478.5） | 判定 |
|---|---|---|---|---|---|---|

| R15 | 基线家系 v5+v6 配方+proxy 三通道 | v5，230/230（best 221） | ep221 | .8872/.8551、FP43(10.43/h)、P95 2148、early 0 | 未跑 | 否决（通道方向耗尽/四点单轴闭环） |
| R16 | 链软目标蒸馏（单模型） | v5，~ep60 停止（用户降级） | — | — | — | 降级留档（方向⑤基座） |
| R17 | 方向①：e147 init+冻结前端+锚定 | v5，早停 ep22（best 7） | ep0/ep7 | ep0 .9566/.9339、FP45(10.94)、**P95 1961（首个非融合击穿）**、early0；ep7 .9544/50/2144/early1 | 未跑 | 否决（B1/A1） |
| R18 | 方向②：真实 AEC 全分支一致 | 四槽 materialized+APM sidecar，早停 ep186 | ep156 | .9479/.9237、FP91(23.16)、P95 2595、early2、B3 .8333 | 未跑 | 否决（P7 veto） |
| R19 | 方向③：跨时间相关特征 | v5 2ch，用户叫停 ep193 | ep171/ep193 | ep171 .9046/53(13.0)/1937/early0/B3 .7868；ep193 .9328/59/2140 | 未跑 | 否决（冗余观测） |
| R20 | 方向④前半：弱标签清理 | v8r7，早停 ep222 | ep192/ep171 | ep192 .9588/144(38.0)/2009/early1/B3 .8333 | 未跑 | 否决（同轴轻移） |
| R21 | 方向⑤：链软概率蒸馏 | v5，早停 ep20 | ep0/ep20 | ep0 .7701/21(4.96)/2615；ep20 .9414/38/2172（=基线） | 未跑 | 否决（软目标不适用） |
| R22 | 方向⑤：链决策蒸馏 | v8r7，早停 ep15 | ep0/ep15 | ep15 .9566/58(14.13)/2168/early0/B3 .7032 | 未跑 | 否决；veto 迁移最强（42.6→14.1） |
| **R23** | 方向⑤：机制组合（R17 ep0+决策锚） | v8r7，20ep | ep12 等 | .9544/.9314、**FP29(6.88, U95 9.38)**、P95 2172、early0、B3 .7032；A1 7 flips | **全量**：heard .8851、**FA 14.80**、heardP95 2792、drain 2794 | **PASS A3/B1/B5/C1/C2**（历史最强画像之一） |
| R24 | R23+onset0.30 | v8r7，20ep | ep8 | .9523/6.88/2144/B3 .7032 | 未跑 | 否决（B2/B3 钉死） |
| R25 | 从零+决策锚+onset0.3 | v8r7，早停 ep43 | ep13–42 | ep28 .9610/15.76/2222/B3 .7438；ep42 .9631/28.5/2396 | 未跑 | 否决（B1↔B3 单轴映射） |
| R26 | R25ep28+锚权1.0 | v8r7，12ep | ep6/ep11 | ep6 .9610/13.05/2384/B3 **.7868** | 未跑 | 否决（锚饱和） |
| R27 | R23ep12+锚权0.2（延迟） | v8r7，8ep | ep2/5/7 | .9544/9.10/2144/B3 .7032 | 未跑 | 否决（只换 FP） |
| **R28** | 方向④后半：难负样本 | v8r7，10ep | ep5/ep9 | ep9 .9501/.9262、**FP24(5.67, U95 7.97)**、P95 2172、B3 .7032 | **全量**：heard .8621、**FA 9.87**、heardP95 2798、drain 2801 | **PASS A3/B1/B5/C1/C2**（最佳候选画像） |
| R29 | 概率温度 1.5 探针 | 免训练 | — | P95 2172→2148（−24ms） | — | 校准路线否（证据受限） |
| **R30** | 延迟：onset-aligned 决策目标+onset0.3 | v8r7，10ep | ep2/5/9 | ep5 .9544/14.66/2144/B3 **.7438** | **全量**：heard .8736、P50 431、**P95 2345（−453ms）**、drain 2352、FA 29.6 | PASS A3/B3/C1/C2（时序修复机制验证） |
| R31 | R30+HN0.4/hard-far0.8 | v8r7，10ep | ep5 | .9523/**9.56**/2148/B3 .7438 | 未跑 | 部分（延迟/FP 平衡点） |
| R32 | R31+onset1.0 | v8r7，6ep | ep1/3/5 | ep1 .9544/20.75/**2057**/B3 .7438 | **全量**：P95 2720、FA 34.5 | 否决（合成域加压恶化 e2e） |
| R33 | R31+focal 正帧 γ=1 | v8r7，8ep | ep2/5/7 | .9479/10.0/2298/B3 .7438 | 未跑 | 否决 |
| R34 | R31+中SER带噪正例上采样 | v8r10，6ep | ep1/3/5 | .9523/9.8/2148/B3 .7438 | 未跑 | 否决（证据受限非曝光受限） |

---

## R0 基建详情（2026-10-12）

**交付**：`scripts/render-aware-dtd/rounds/`（随仓库提交）：排除清单生成器 / 训练守护 / 训练监控 / 验证集选点 / 导出+声学评测（隔离运行时）/ 环境稳定性校验 / e2e+compare / 达标判定，共 8 件；用法与约定见该目录 README。`eval-exclusions.json` 与锚点 `assets/anchor-e147-validation.json` 一并冻结。

**自证（candidate = 基线模型 e147，零改动）**：
1. 排除清单：378 device case 源文件全部可解析（0 missing）；继承重叠 96 文件与人工盘点一致；e2e 资产重叠分类（speech 63 训练摄入 / 64 仅冻结测试池；rir 96/12；noise 0）。
2. 选点：从 candidate-01 历史 204 epochs 独立复现 e147 为唯一 policy 合格点。
3. 导出：重导 e147 ONNX 与打包基线 **逐字节一致**（`3EB35D15…`）。
4. acoustic 全量重放：TP/FN/FP/TN、recall、FP/h（点+U95）、P50/P95 与基线**逐字段一致**；`--acoustic-only --enforce-comparison` = Passed。
5. e2e 重放两次：决策级/门禁级指标与基线完全一致；CER（0.4625→0.4663→0.4684）与部分小切片停止延迟运行间非确定，**同模型接连两次重放即产生 20 条 e2e-slice CER finding**。
6. 判定链：对 no-op 候选判 FAIL（A/C 过、B 无改善），行为符合设计。

**协议修订（据 1-6 自证）**：达标定义 A 由"compare 退出码 0"修订为"A1 声学层 strict compare Passed（硬门）+ A2 完整 compare 无严重回退（e2e-slice 噪声豁免清单）+ A3 零错误"；B/C 项全部使用已证确定的指标。

**附带代码修复**：`ComparisonProgram.cs` 的 `--acoustic-only` 模式此前必抛异常（置空 E2E 路径后仍被无条件要求），已最小修复；只影响此前不可用的模式，不改变完整比较行为与既有基线证据。

**预登记风险实际走向**：不适用（基建轮）。

**结论与下一步**：工具链可用，R1（v8 设备族覆盖）可以启动；R1 立项文件需引用 `eval-exclusions.json`（96 文件继承重叠保留、新语料整 session 排除）。

---

## R1 详情（进行中，2026-10-12）

**立项**：`artifacts/dtd-rounds/r1/round.json`（假设/唯一变量/配方/预登记判据与风险）。

**数据集构建（已完成并验证）**：
- 池扩展：v5 real_train（13,759 行，逐字节保留）+ 7 个未覆盖语料的 12,466 新行，整池筛查剔除 87 条退化行（哑音 far 对/无有效近端标签），最终 26,138 行（far 16,552 / near 9,586）。
- 评测隔离：评测占用 session 整段排除、新增行零评测文件重叠；96 文件继承重叠保留（基线条件）。
- 物化生成：`prepare_materialized_v5.py`（冻结原件、seed 43）→ 48,000+1,500，0 失败；验证集从 v5 复制（逐位一致）。
- 逐位对拍（`dataset/verify-report.json` = PASS）：21 个 manifest 全等；26,425 个"池触及"行与分支决策逐一对应；19,520 个非 cold-onset 未变化行 metadata+音频全等；验证集 1,500 音频全等；96/96 继承重叠在池。唯一已知差异 = cold-onset 对齐 v4 生成代码版本差（2,054 行，与池扩展正交，A–H 轮数据集同样如此）。
- 代码等价自证（`dataset/e147-reproduce-report.json` = PASS, diffCount 0）：当前训练/评测代码 + e147 权重在 v8r1 验证集上精确复现 candidate-01 的历史验证记录（含全部 strata）。
- 覆盖率（48,000 训练样本实际用新语料的行数）：far mobile 5,630 / fullband 2,498 / icassp2022 405…；near mobile 1,840 / fullband 1,383…

**训练**：230/230 完成（无早停），trainer best_epoch=224。

**选点（P4）**：anchor 严格非回退 = **0 个 epoch**；anchor_nearest 池 = 0；选点 e224（早停 best）/ e213（AP best）。验证集上 e224/e213 全轴劣于 anchor（AP/F1/recallL95 微降、fpU95 57.9/58.6 vs 47.35、P50 220）——机制为旧族曝光减半（稀释），新族收益在 v5 验证集不可测。
**明示偏离**：方案 P4 的"验证集被 anchor 严格支配 → 免评测 veto"前提对 R1 不成立（验证集不含新族，R1 的收益轴在其上不可测；预登记 R1 成功判据 P7 本就以 benchmark 为准），故继续 benchmark 评测（成本 ~25 分钟）。记录于 round.json.selection.deviation。

**声学评测（P6/P7）**：
| | 基线 | e224 | e213 |
|---|---|---|---|
| TP/FN/FP | 434/27/33 | 430/31/33 | 430/31/35 |
| recall / L95 | 0.9414/0.9161 | 0.9328/**0.9061** | 0.9328/0.9061 |
| FP/h 点/U95 | 7.903/10.568 | **7.864/10.514** | 8.361/11.086 |
| P50/P95 | 396/1973 | 386/1973 | 386/1973 |
| noisy-movement | 26/30 | **27/30** | — |
| 官方 acoustic-only compare | — | **Regressed（207 findings）** | Regressed（228） |

**FP/FN 拆解（e224，配对）**：修好 15 FP、新增 15 FP；修好 7 FN、新增 11 FN。
- **首次出现"目标族被压低"**：icassp2022 18→13（−5）、mobile 6→3（−3）——设备族覆盖机制方向正确；
- 代价一：**test_set_noisy 7→14（+7）**（不可训练族，评测整占）；
- 代价二：召回 −0.87pt（旧族稀释），新 FN 扩散于 mobile/real/fullband/icassp；
- 意外：test_set_clean FN 5→1（−4）。

**判定：否决**（A1/B2/C1 未达；B1/B3 勉强达标）。未消耗 e2e（P7 veto）。
**结论**：设备族覆盖是首个在目标维度见效的数据干预（与 A–H 八轮的"修 N 引 >N"形状同构但首次修好了目标族），但 net 收益为零；召回回退指向曝光稀释。→ **R2 = 复权重（0.20→0.40）恢复旧族曝光**。

## R2 详情（2026-10-13，否决）

**唯一变量**：`--aec-real-weight 0.20→0.40`（复用 v8r1，无重训）。230/230 完成。
**选点**：e208（唯一选点）。验证集首次出现工作点与 anchor 等高：fpU95=47.35113126087358 与 anchor **完全相同**、P50=200 相同、recall@0.1%FPR 0.4738 反超 +0.011；仅 AP/F1/recallL95 微低。仍 0 个严格非回退 epoch → 延续 R1 的明示偏离继续 benchmark。
**声学**（官方 acoustic-only：**Regressed 290 findings**）：

| | 基线 | e208 |
|---|---|---|
| TP/FN/FP | 434/27/33 | 403/**58**/**21** |
| recall / L95 | 0.9414/0.9161 | 0.8742/**0.8408** |
| FP/h 点/U95 | 7.903/10.568 | **4.946/7.122（-37%）** |
| P50/P95 | 396/1973 | 386/**2144** |
| 早触发 | 0 | **1** |
| noisy-movement | 26/30 | 25/30 |

**配对拆解**：FP 33→21（修好 19，icassp 18→10、mobile 6→2、fullband 1→0；新增 7）——**项目史上首次 FP 大幅净降**；FN 27→58（修 2 引 33，real_doubled +9、mobile +7、icassp +7…）。
**判定：否决**（召回崩塌、P95 恶化、新早触发；A1/B2/B3/C1 未达）。未消耗 e2e。
**联合结论（R1+R2）**：real 曝光配比是单一"精度↔召回"轴——R1(w0.2)=(33 FP, .933 recall)、R2(w0.4)=(21, .874)、基线(w0.2, v5 池)=(33, .9414)。v8r1 家系两个工作点均被基线支配；基线在这条轴上的位置本身接近最优。→ **R3 = 工作点重选**：花 R2 的 FP 余量换回召回（阈值 0.55→0.48，验证集预登记扫描选中，规则与数据见 `r3/selection/threshold-sweep.json`；若否决则运行时阈值回退 0.55）。

## R3 详情（2026-10-13，否决）

**方法**：R2 e208 权重不动；验证集唯一扫描（0.42–0.55 六档，预登记规则运行前固定）选中 **t\*=0.48**（最高 T 满足 recallL95≥0.7131 且 p50≤200，见 `selection/threshold-sweep.json`）；`RenderAwareDtdOptions.Threshold` 临时改 0.48 后 benchmark 一次评测定判。
**结果**：召回仅 +0.65pt（0.8742→0.8807，TP 403→406）、FP 21 持平、FP/h 4.946 持平、P50 386→380、**P95 2144 完全不变**、早触发仍 1。
**判定：否决**（C1/B2/B3/A1 未达）。未消耗 e2e。**运行时阈值已回退 0.55**、DLL 已重建（源码树恢复冻结状态）。
**机理**：判决层对阈值杠杆不敏感（策略平滑/窗口/累加器主导；验证集预期 +2.3pt 实际 +0.65pt），且 P95 由特定尾部 case 决定，阈值不可及——与 E–H 轮"累加器已在最优点"独立印证。

## R1–R3 综合结论（自动轮次暂停点，2026-10-13）

- **v8r1 实验族的前沿已被完整测绘**：三个工作点 (FP 33, recall .933) / (21, .874) / (21, .881) 全部被基线 (33, .9414) 支配；训练侧（权重）与判决侧（阈值）两条轴都无法同时满足 C1 与 B1。
- **唯一项目史上首次重大单项突破**：FP 净降 -37%（R2/R3，修 19 引 7）；但以召回 -6pt 为代价，且 B2（P95）与 B3（noisy 切片）在所有点上未动/恶化。
- 按方案自适应规则，**"设备族覆盖"维度已耗尽**（R2/R3 连续两轮 net 负 churn）；剩余 backlog 项（真实双讲段 R5 / 契约扩展 R6）均为大额投入或需用户批准 → **暂停自动轮次，向用户呈报决策**（选项：R5 真实双讲弱标签 / 二次确认阶段（近契约扩展）/ 接受低误报档为产品选项 / 调整判据口径）。


## R4 详情（2026-10-13，否决但重大收获）

**唯一变量**：无训练——二次确认/否决融合（base=candidate-01 e147 + gate=R2 e208，soft-min lo=0.05，单 ONNX 同契约，状态 [6,1,288]）。
**验证集预登记分析**（`r4/selection/veto-rule-analysis*.json`）：soft-min 档 lo=0.05 零 TP 损失（hard-gate 全不可行）。
**benchmark 实测**：**FP 33→24（修 9、新增 0 —— "只删不增"结构性质在真实管线成立）**、FP/h −28%、早触发 0；但否决 6 个 TP（validation 零损失→评测集外推差再现）→ standalone 否决（A1/C1 未达）。
**遗留资产**：`export_fused_veto_onnx.py`（双模型融合导出+parity）、`swap_and_eval.ps1`（任意 ONNX 换入隔离运行时评测）；坑：fused contract 的 `model_type` 必须保持 `render-aware-nearend-dtd`（运行时硬校验）。

## R5 详情（2026-10-13，否决；融合空间大幅推进）

**唯一变量**：`--onset-loss-weight 0.10→0.30`（数据集=v5 与 candidate-01 全同）。219/230 早停（patience 30），best=188，guard attempt 1。
**选点**：无 policy 合格；**e165 = 十一轮来首个验证集对 anchor 三轴不劣的工作点**（rL95 0.7211 ≥ 0.713075、fpU95 47.351 精确相等、p50 200 相等）；另有 e188（早停 best）、e191（AP best）。

**standalone 声学**（均否决；机制作用域重定向）：

| | 基线 | e165 | e188 | e191 |
|---|---|---|---|---|
| TP/FN/FP | 434/27/33 | 439/22/38 | 435/26/51 | 437/24/60 |
| recall / rL95 | .9414/.9161 | **.9523**/.9288 | .9436/.9186 | .9479/.9237 |
| FP/h 点/U95 | 7.903/10.568 | 9.181/12.034 | 12.52/15.81 | 14.90/18.48 |
| P95 | 1973 | 2132 | **1969** | 2045 |
| early | 0 | 1 | **0** | 1 |
| noisy-move | 26/30 | **28/30** | **28/30** | **28/30** |

**onset 0.30 的真实作用域**：不是基准 P95（唯一摸到 1969 的是 e188，仅 −4ms）而是**召回 +1.1pt（历史最高 .9523）、noisy-move 28/30（B3 首次达标）、e188 基座 early 归零**；FP 随 AP 单调上扬，须判决层承担。**新工具**：`analyze_union_rule.py`（并集"只增"规则验证集分析）、`export_fused_union_onnx.py`（union 导出，支持第三模型 veto 链）。

**融合矩阵（benchmark 声学，全部新证据）**：

| 配置 | TP/FN/FP | recall/rL95 | FP/h 点/U95 | P95 | early | noisy |
|---|---|---|---|---|---|---|
| veto e165×R2 lo0.05 | 431/30/31 | .9349/.9086 | **7.416/10.008** | 2132 | 1 | 28/30 |
| veto e188×R2 lo0.05 | 428/33/38 | .9284/.9012 | 9.167/12.016 | **1969** | 0 | 28/30 |
| union e147∨0.6·e165 | **440/21/35** | **.9544/.9314** | 8.407/11.146 | 1973 | **0** | 27/30 |
| union+veto R2 lo0.05 | 432/29/26 | .9371/.9111 | **6.177/8.57** | 1973 | 0 | 27/30 |
| **union+veto R2 lo0.02** | **437/24/33** | **.9479/.9237** | 7.904/10.569 | 1973 | **0** | 27/30 |

**结论（同轴律再现）**：union（只增）拿到召回/难切片/early；veto（只删）拿到 FP 深降；3 模型链 lo=0.02 = **全轴 ≥ 基线（FP 33=33 精确持平）+ 召回 +0.65pt + rL95 +0.76pt + B3 + early 0** —— 距达标仅剩 **B1 严格**（需净删 ≥1 FP 且零代价）与 **B2 严格**（需 P95 < 1973）。**明示偏离**（round.json.fusionExperiments.deviations）：union 资格集空后以近轴点 s=0.6 续测；veto lo=0.05 破 C1 后降档 lo=0.02 复测一次。
**e2e+compare（机器判定，`r5/acceptance.json`）**：union+veto lo=0.02 链全量评测——**PASS 5 项（A3/B3/C1/C2/C3），FAIL 6 项**：A1 Regressed（**69 findings = 5 case + 64 slice，历史最少**，对照 R1=207、R2=290）；A2 severeCount 69；B1 fp/h 差 0.001（FP 计数相同）；B2/B4/B5 精确持平（1973=1973、1475.0=1475.0、29.599=29.599）。5 个 case 回退 = 2 新 FP（fullband_mobile farend、icassp farend）+ 3 新 FN（real/real_doubled/icassp doubletalk_movement）。e2e 层 18 条 finding 全在噪声豁免清单内（CER/audibleTail/stopP95），与 R0 同模型重放噪声同量级；heard recall/drain P95 与基线完全相等。**工具链扩充**：run_e2e_and_compare.ps1 增 -RuntimeName/-AcousticDir（融合候选非 epoch 命名）、check_acceptance.py 增 --acoustic-dir。**判定器 bug 修复**：check_acceptance.py 的 B3（召回升高）误用 strict_lower 方向 → 新增 strict_higher 修正；修复后 B3 PASS（旧轮次无结论翻转，均已被其他项否决）。

## R6 详情（2026-10-13~14，否决；机器判定 6/9 历史最佳）

**训练**：onset 0.30 × v8r1，230/230 全训（无早停，best 213），7.6h。final val AP 0.96556。
**选点**：anchor_nearest **e144**（rL95 0.7196 ≥ anchor 且 fpU95 46.02 **严格低于** anchor 47.351——首次验证集双轴严格更优）；e213（trainer best，召回差，弃）。

**standalone 声学**：e144 = TP 432/FN 29/FP 30，recall .9371/rL95 .9111，**FP/h 7.142/9.686（模型级首次 B1 双降）**，P95 2144，early 1，noisy 27/30。组合假设验证：v8r1 设备族 FP 抑制（icassp 18→15）与 onset 的 noisy 增益并存；代价 = 召回 −0.43pt、尾部更慢。

**R6 链（union(e147, 0.6·e144) + veto(R2 e208)）**：

| | 基线 | lo=0.05 | **lo=0.02** |
|---|---|---|---|
| TP/FN/FP | 434/27/33 | 430/31/25 | **434/27/32** |
| recall/rL95 | .9414/.9161 | .9328/.9061 | **.9414/.9161（逐位相等）** |
| FP/h 点/U95 | 7.903/10.568 | **5.930/8.282** | **7.653/10.279** |
| P95 | 1973 | 1973 | 1973 |
| early | 0 | **0** | **0** |
| noisy-move | 26/30 | **27/30** | **27/30** |

**机器判定（r6/acceptance.json）**：**PASS 6 项（A3/B1/B3/C1/C2/C3）、FAIL 5 项**——A1 Regressed 67 findings（**4 case：1 新 FP + 3 新 FN**，历史最少）；B2 1973=1973；B4 1475.0=1475.0；B5 29.599=29.599（**e2e 层决策零扰动——R5/R6 两代链与基线逐位相等，与 acoustic 层解耦**）。
**结论**：B1 首次以"召回与基线逐位相等"形态通过；剩余失败全部同型（4 case 翻转 + 3 个精确持平），指向结构墙——A1 需 veto 删除集 ⊆ FP 集（实测每次伤 TP）、B4/B5 需 e2e 域内干预。→ **R7 = TTS 渲染 far 域**（B4/B5 域偏移假设的直接干预）。

## R7 详情（2026-10-14，否决但打开 e2e 域；含判定器 bug#2 修复）

**假设**：e2e 套件 far-end = VoxCPM TTS 渲染（echo-only 假接受为基线的全部 6 例）；训练 far 域从未覆盖 TTS 渲染声 → e2e 层对前代所有候选零扰动即域偏移证据。

**构建**：`synthesize_tts_corpus.py`（240 条中英混合，同部署包 int8、seed 23+i）→ `build_v8r2_tts_pools.py`（farend_train 120,251→143,951 行，TTS 段份额 16.5%；其余 manifest 逐字节复制）→ 冻结 `prepare_materialized_v5.py`（seed 43）+ **验证集从 v5 镜像（3,000 文件 0 差异）**。首条训练样本即 `far:wenet_rir_echo:tts-online` ✓。

**训练/选点**：230/230 全训（无早停，best 220，~6.7h）。**无 anchor_nearest 合格点**——TTS 域训练姿态突变：验证集召回大涨（e220 rL95 .7262、e213 .7335 历史最高档）同时 fpU95 大涨（69.0/72.9）。

**standalone 声学（benchmark）**：e220 = TP 425/FN 36/FP 30，recall .9219，**fp/h 7.135/9.678（B1 双降）**，P95 2009，noisy 27/30；e213 = FP/h 6.89/9.39（FP 最低档）但召回 .9132。均为"低 FP+低召回"姿态。

**链（union(e147, s·e220)+veto(R2)）**：验证集选中 s=0.5（唯一近轴档）；lo=0.05 = FP/h **5.685/7.995（链族历史最深）**但 recall .9284、noisy 26/30；lo=0.02 = 7.404/9.992、recall .9349、noisy 26/30——声学被 R6 链支配。

**e2e 层——本轮核心发现（3 个实验）**：
1. **R7 链（s=0.5, lo=0.02）e2e 与基线逐 case 逐位相同**（164 例全部，含 ms 级 detectionTimeMs）。归因：s=0.5 → 激活门槛 0.55/0.5=1.1，**增量成分在 e2e 层完全不开火**，链 ≡ e147。
2. **e220 standalone e2e（决定性实验）**：**heardPositive 74→80**（censored 13→7，全部 6 个此前未 heard 的正例被捕捉）、**echoOnlyHeard 6→12**、heard P95 1475→1595、**falseAccept 29.6→64.1**、stopP95 1713→3443。→ **证明 e2e 层确实经过换入的 DTD**（项目首次让 e2e 层移动）；TTS 域假设成立但敏感度增益与误听同升。
3. **s=1.0 链（veto 全开）**：heard 79（C2 .908 首次提升）、但新误听 9 例全为 `tts-echo-*` 负例（SER −12~+12dB、ERLE 0–24dB、环境噪声/音乐），echoOnly 14、falseAccept 74.0、stopP95 3399 —— **R2-veto 无法分离 TTS 域误听与真 heard**。

**机器判定（acceptance-s100chain.json）**：PASS 3 项（A3/B3/C2）、FAIL 8 项（A1/A2 severe 426、B1/B2/B4/B5/C1/C3）。**判定器 bug#2**：C3（drain P95 延迟"不降级"）误用 not_lower → 新增 not_higher 修复（修复后 C3 正确 FAIL；旧轮次因精确相等无结论翻转）。

**结论与下一步**：TTS 域是 e2e 层的真实杠杆（首次可动），当前缺口 = 真 heard 增益与 tts-echo 误听未分离。→ **R8 = TTS 份额 16.5%→33%**（强化回声抑制）；**R9 = 多音色语料**（用户指示 Qwen3 音色丰富；vc237+q3-70 已备），**R9 最终训练参数在 R8 结果分析后确定**（用户 2026-10-14 指示）。

## R8 详情（2026-10-14~15，否决；TTS 剂量轴反噬确证）

**假设**：R7 证明 TTS 域曝光是 e2e 层唯一可动杠杆（realized 样本份额 11.4%）。剂量翻倍（行份额 16.5%→33.0%，multiplicity 100→250）以强化 TTS 回声抑制，直指 B5 交换率。

**训练/选点**：v8r3（唯一变量 = TTS 行份额；其余 manifest 逐字节复制、验证集 v5 镜像 0 差异）。223/230 早停（patience 30），best e192，valAP 0.96536（全程健康爬升，无过拟合）。选点：**e185（anchor_nearest：验证集 rL95 .7174 > anchor .7131 且 fpU95 46.69 < 47.35 双轴严格更优）**、e192（早停 best）、e196（AP best）。

**standalone 声学 — device 族双讲塌陷（三点同型）**：

| | 基线 | e185 | e192 | e196 |
|---|---|---|---|---|
| recall / rL95 | .9414/.9161 | **.8134**/.7753 | .8091/.7707 | .8351/.7985 |
| FP/h 点/U95 | 7.903/10.568 | 5.916/8.262 | 6.399/8.824 | 5.914/8.260 |
| noisy-move | 26/30 | 26/30 | 27/30 | 27/30 |

塌陷切片：android_mobile 双讲 .78→.48、windows_pc .93→.70、device_fullband .96→.67、device_fullband_mobile .78→.48、device_real_doubled .96→.57、icassp2022 1.00→.86；**未坏**：device_real、test_set_clean/noisy、synthetic。即受损全部集中在 device-session 族与 doubled/icassp 的双讲切片，训练覆盖过的 device_real 与 test_set 系列无损。

**控制实验（协议完整性）**：e147（打包基线）经同一管线当晚重放 = 与 10-12 基线**逐位一致（overall 0 差异 / 58 切片 0 差异）** → 二进制与评测链无漂移，塌陷是真实模型行为（非管线回归）。

**单一变量核验**：v8r2 vs v8r3 逐样本标志分布逐项一致（movement 21531、hard_far 11466、render_gap 8572、nearend_noisy 21514、ser_mean 12.44、device profile 计数全同、mode 计数全同）；唯一差异 = TTS far 样本份额 **11.4%→21.3%**。剂量-反应成立：11.4%（R7：正常）→ 21.3%（R8：塌陷）。v5 镜像验证集对塌陷**盲**（e185 验证集姿态与 anchor 可比）。

**链（union(e147, 1.0·e185) + veto(R2 lo=0.05)）**：recall .9328/rL95 .9061、FP/h **7.649/10.275**（链族历史最深）、noisy 27/30 —— 被 R6 链支配（R6: recall .9414 逐位相等 + 7.653/10.279）。

**e2e 双层协议**：
1. **e185 standalone**：heard 75/87（.8621）、censored 12、echoOnly **11**、FA/h **59.2**、heardP95 1626、stopP95 1761、drainP95 1638、early 4。对比 R7 e220（heard 80、echoOnly 12、FA 64.1）：**heard 增益仅 +1（R7 为 +6）**——剂量翻倍只买到了伤害的大头，分离能力未增强。
2. **s=1.0 链**：heard **74（≡基线）**、echoOnly 14、FA/h 74.0、heardP95 1598、stopP95 2931 —— 劣于 R7 s=1.0 链（heard 79、C2 .908）；echo-only 误听集合与 R7 链共享 ~13/15（增量仅换掉 e2e-echo-only-26/56 → 62/63）。

**机器判定**：standalone `acceptance.json` **FAIL 8/11**（PASS A3/B1/C2；severe 557）；链 `acceptance-s100chain.json` **FAIL 6/11**（PASS A3/B1/B3/C2；severe 416）。

**结论**：TTS 剂量轴 = **反噬**（零 e2e 净增益 + 声学灾难）；R6 链仍为前沿。→ **R9 = 多音色内容 @ R7 匹配剂量**（vc237+q3-70 共 307 条 ×80；行份额 16.96% ≈ R7 同档 16.46%，realized 预计 ≈11.8%）：单变量换内容轴，检验音色多样性是否能在保住声学姿态的前提下改善 e2e；若不成立则 TTS 轴（剂量×内容）耗尽，转下一 backlog 方向。旧 v8r4（mul 190，与 R8 同剂量）据 R8 结论重建为 mul 80，构建报告留档 `r9/dataset/pool-build-report-mul190.json`。

> **归因修订（R9 结案后，2026-10-15）**：R9 在同剂量（realized 11.64% ≈ R7 11.4%）下复现了与 R8 同型的 device 族双讲塌陷 → 上段"剂量-反应"归因**被修正**：R8/R9 的共同点不是剂量而是"TTS far 域扰动"（换剂量、换内容、换抽签任一扰动都可能触发；R7 属未崩的单次抽签）。机制（内容/重抽/流耦合）未能分离——`single_variable_audit` 见 `r9/round.json`（基础 manifest 全等、聚合构成全等、1,433 样本字段错位）。


## R9 详情（2026-10-15，否决；多音色假设失败，TTS 轴判耗尽）

**假设（据 R8 分析定参）**：R8 的剂量翻倍反噬 + R7 剂量被证有效 → 换内容轴单变量：TTS 语料 单音色 vc-237 → 多音色 vc237+q3-70=307 条（用户指示 Qwen3 音色丰富），剂量回退到 R7 档（multiplicity 80，行份额 16.96%，realized 11.64% ≈ R7 11.4%）。检验音色多样性能否在保住声学姿态的前提下改善 e2e 分离。

**训练/选点**：v8r4（重建；0 失败；验证集 v5 镜像 3000 文件 0 diff）。230/230 全训（无早停），best valAP **0.9659**（ep182，历史最高档）。选点：e170（anchor_nearest：rL95 0.713075 ≈ anchor 恰同值 + fpU95 44.02 < 47.35）、e220、e182（rL95 .7386 验证集召回档历史最高）。

**standalone 声学 — 同 R8 型塌陷（三点全崩）**：

| | 基线 | e170 | e220 | e182 |
|---|---|---|---|---|
| recall / rL95 | .9414/.9161 | .8373/.8009 | .8590/.8243 | .8959/.8647 |
| FP/h 点/U95 | 7.903/10.568 | 5.679/7.987 | 6.893/9.399 | 7.884/10.542 |
| noisy-move | 26/30 | 27/30 | 26/30 | 26/30 |

**链（union(e147, 1.0·e170) + veto R2 lo0.05）**：recall .9328/rL95 .9061 与 R8 链相同，但 FP/h **8.634/11.401 高于基线**（B1 首次在链上失守）——被 R8 链、R6 链双重支配。

**e2e 双层协议**：
1. e170 standalone：heard **73/87（.8391）——首次跌破基线 74**（R7 e220 为 +6）；censored 14、echoOnly 12、FA/h 64.13、heardP95 1584、stopP95 2710、drainP95 1596。
2. s=1.0 链：heard 73、echoOnly 14、FA/h 73.998（与 R7/R8 链**逐位相同**——链结构下 echo-only 误听取决于 keep/veto 骨架与套件，14 例为结构性）、heardP95 1520、stopP95 2060。

**机器判定**：standalone `acceptance.json` **FAIL 8/11**（PASS A3/B1/B3；severe 587；**C2 首次被跌破**）；链 `acceptance-s100chain.json` **FAIL 8/11**（PASS A3/B3；severe 518；B1 失守）。

**单一变量审计（`r9/round.json.single_variable_audit`）**：20/21 基础 manifest 在 v8r1/v8r2/v8r3/v8r4 四个 src 根**逐字节一致**；聚合构成（mode/far/near/flag 计数、ser 均值）全等；差异 = q3 内容 + 136 个 TTS 增量样本 + **1,433 个样本的 device/ser/nonlinear 字段错位**（RNG 流耦合）。语料音频质量核验通过（vc/q3 RMS、时长、直流均正常）。

**结论与修订**：多音色内容假设全面失败——匹配剂量下仍触发同型塌陷，e2e 层还首次跌破基线。结合 R8：**TTS far 域训练对 device 族双讲召回具有不可控的不稳定性**（R7 属未崩的单次抽签）；R8 的"剂量-反应"归因相应修订（见 R8 节注）。按方案 §5"连续两轮 net 负 churn → 维度耗尽"，**TTS 域轴判耗尽**。→ **R10 = 容量叠加**（backlog 最高优先项：gru-layers 3→4 + frequency-bands 80→128，组合不可拆照 H 轮先例；× R6 配方 onset 0.30 × v8r1）。注：训练器 argparse 白名单原不含 4/128（`choices=(2,3)`/`(48,64,80)`），已最小扩展（模型代码本就全参数化）；R10 首启因白名单报错 30 秒即退，重试后正常。


## R10 详情（2026-10-15，否决；容量轴 + B2 墙结构性归因）

**假设**（据 R8/R9 结论定方向）：TTS 域轴耗尽后转 backlog 最高优先项——容量叠加（旧 H 轮验证集最强方向）：gru-layers 3→4 + frequency-bands 80→128（组合不可拆，照 H 轮先例），× R6 配方（onset 0.30 × v8r1，seed 43）。目标：设备族召回稳健性 + B2 P95 尾部 + A1 case 翻转。

**工具链**：首启因训练器 argparse 白名单不含 4/128 报错（模型代码本已全参数化）→ 最小扩展 `train_dtd.py` choices；链导出 `export_fused_union_onnx.py` 的 state 切片原假设成员层数相同 → 改为按各模型自身层数切片（3+4+3=10 层，state [10,1,288]）。

**训练/选点**：v8r1（验证完好）。230/230 全训，best valAP **0.9661**（ep213）。**anchor_nearest 池=3**（历轮首次多点合格）；选点 e157（rL95 .71599 ≥ anchor .71308；**fpU95 47.35113126087358 恰等于 anchor**；P50 相同）、e205、e213（rL95 .7357 验证档最高）。

**standalone 声学**：

| | 基线 | e157 | e205 | e213 |
|---|---|---|---|---|
| recall / rL95 | .9414/.9161 | .9371/.9111 | .9306/.9037 | .9371/.9111 |
| FP/h 点/U95 | 7.903/10.568 | **7.383/9.964** | **6.896/9.403** | **7.637/10.258** |
| P50 / P95 | 396/1973 | **368**/2132 | 368/2410 | **362**/2144 |
| early | 0 | 1 | 1 | 1 |
| noisy-move | 26/30 | **28/30（rL95 .7868）** | 26/30 | 27/30 |

**三点全部 B1 双降 + P50 改善（−28~−34ms）+ e157 取得历史最强 B3（28/30）**；但 C1（rL95 −0.5pt）、B2（P95 2132+）、每点新增 1 早触发。

**融合**：链 s=1.0 lo=0.05 = recall .9393/FP 8.388/11.121；lo=0.10 = recall .9328/FP 7.890/10.550 —— **两档均被 R6 链支配**（后者 recall 逐位相等 + FP 7.653/10.279 + early 0 + P95 1973）。

**e2e 双层**：e157 standalone heard 75（+1）、**echoOnly 18 / FA/h 93.7（standalone 最差）**；链 heard 75、**echoOnly 20 / FA 103.6（链族最差）**。注意：v8r1 不含任何 TTS 训练数据，e2e 误听仍随模型敏感度上扬——B5 与「模型对新 far 域材料的敏感度」同向，不是 TTS 特有。

**机器判定**：standalone `acceptance.json` **FAIL 8/11**（PASS A3/B1/B3/C2；severe 492）；链 `acceptance-s100chain.json` **FAIL 8/11**（PASS A3/B3/C2；severe 482）。

**环境事件与对照（重要）**：当日 09:53（R9 e2e 之后、R10 构建之前）.NET SDK 自 10.0.401 升级至 10.0.402（VS workload 更新），稳定性门对 R10 e2e 报漂移。**双重对照证明行为中性**：e147 重放（10.0.402 构建）声学层与 10-12 基线**逐位一致**（0 overall / 0 slice）；e2e 决策级逐位一致，仅 CER 与 stopP95 两项落在 R0 已证噪声豁免清单内。R10 e2e 以 `-SkipEnvironmentStability` 执行（明示偏离）；对比基线证据仍用 10-12 原件（已证可复现）。

**B2 墙结构性归因（本轮副产物，关键）**：P95 尾部 = **固定 4-case 集合**（synthetic-353/64/44/72，SER −5~+2dB），**三个不同模型（基线/R10 e157/R6 链）的延迟逐毫秒相同**（2652/2560/2360/2072）且峰值概率 0.998–0.999——决策时刻由材料/累加器结构钉死，DTD 侧只能微移 shoulder（e157 把 synthetic-72 削了 16ms，同时 P50 −28ms/P90 −65ms，但 +1 censored 反将官方 P95 推高至 2132）。→ **B2 的"严格下降"判据对纯 DTD 干预的敏感性极低**（需 shoulder 整体左移且 censored 不增；R5 e188 的 −4ms 是历史最好）。

**结论**：容量轴拿到 B1/B3/P50 三线齐进（B3 历史最强），但以 recall/P95/early/e2e 为代价，融合无法回收 → 否决。B2（及 e2e B4/B5 的同型结构）需 shoulder 级左移或共识性改进，非单轴可及。→ **R11 = 真实双讲弱标签**（backlog 最后未试数据轴，R5 存档项）。


## R11 详情（2026-10-15~16，突破轮：声学合取首次清空，e2e 墙仍在）

**假设**：backlog 最后未试数据轴——四大真实族（real / real_doubled / real_doubled_fullband / real_doubled_fullband_mobile）的 `_doubletalk_mic + _doubletalk_lpb` 成对录音（~13k 会话「带真实回环参考」）从未入训（双击段无近端标签）。用会话内回波路径拟合做残差弱标签，确定性注入 v8r1 的 mixture 槽位。

**构建**（`label_real_doubletalk.py` + `inject_real_doubletalk.py`）：
- 标注：逐会话互相关求延迟 → far-only 对频域 Wiener（4096-tap）拟合回波路径（中位 ERLE +17.5dB，修复了 NLMS 发散/对齐反置/OLA hann 边缘爆炸三个 bug）→ 双讲残差能量门（3% 峰值 + 2.5×p10 底噪）→ 官方同款 `labels_for` 派生与质量筛（coverage/活跃时长/ERLE）→ 5,437 会话 → 4,817 有段 → **8,434 段**（movement 53%；SER est p50 24.5dB；按 0.09 active-RMS 配平电平）。
- 注入：v8r1 **hardlink 克隆** + 确定性替换 mixture 15%（3,610 行 = 全集 7.5%），四种族均衡；**关键陷阱：不可直接覆写 hardlink 路径（会改共享 inode 毁掉 v8r1），必须 temp+os.replace**。验证：标签复检 coverage diff p50 0.0015、验证集 v5 镜像 0/50 diffs、v8r1 原件 100/100 独立 inode、清单计数 3,610 real-dt 行。

**训练/选点**：v8r5，230/230 全训（best valAP 0.9645 ep174）；**anchor_nearest 池=5**（历轮最多）。选点 e181/e212/e174。

**standalone 声学（e181）**：

| | 基线 | e181 |
|---|---|---|
| recall / rL95 | .9414/.9161 | **.9696（纪录）**/.9497 |
| FP/h 点/U95 | 7.903/10.568 | 42.58/49.08 |
| P50 / P95 | 396/1973 | 378/**1892（B2 首次实质下降）** |
| early | 0 | **0** |
| noisy-move | 26/30 | **29/30（rL95 .8333）** |

设备族双讲切片**全面改善**（注入的目标族）：android_mobile DT 20/22→22/22、fullband_mobile 20/22→22/22、各 +movement +2、real_doubled+mv 23/23。代价 FP 暴涨于 farend-singletalk（回声-only）。

**融合矩阵（本轮测绘，含关键机制发现）**：
| 配置 | recall | rL95 | FP/h | P95 | early |
|---|---|---|---|---|---|
| A：e181 keep+veto(R2,0.05) | .9393 | .9136 | 10.87 | 1892 | 0 |
| B：union(e147∨e181)+veto(R2,0.05) | .9436 | .9186 | 12.68 | 1892 | 0 |
| B-s080（增量置信带 0.6875+） | .9436 | .9186 | 8.90 | 1937 | 0 |
| B-s075≡s070（格点：无增量在该带） | .9284 | .9012 | 5.68 | 1973 | 0 |
| **veto 换 e157（0.05）** | **.9566** | **.9339** | 10.93 | 1937 | 0 |
| **B-s080-e157-lo020** | .9501 | .92625 | **7.406** | **1937** | **0** |

机制：s 格点离散（增量激活按置信带跳变）；veto 模型 R2→e157 显著抬升召回上限（e157 自身高召回低 FP）；lo 滑杆 0.05→0.20 用召回余量换 FP，直至四项全过。

**最佳链（union(e147, 0.8·e181) + veto(e157, lo=0.20)）机器判定**：**PASS 7/11（历史最佳）**——A3/B1/**B2**/**B3**/**C1**/C2 全通过（**声学合取首次清空：B1 7.406/9.996、B2 1937、B3 28/30、C1 .92625、early 0**）；FAIL：A1（severe 169；case 翻转 **13** = 10 新 FP（icassp2022 主导）+ 3 新 FN——比对文件全量口径；判定器 severe 列表截断为 10）、A2、**B4（1675）/B5（54.3）/C3（1677.5）**。
**链 e2e**：heard 76（+2，C2 .8736）；echoOnly 10、FA 54.3（优于 R7–R10 链的 64–104 但仍远高于基线 29.6）；heardP95/drainP95 1675/1677.5——e181 的慢增量拉高了 e2e 尾延迟。standalone e181 e2e：heard 76/FA 54.3/heardP95 1903（两跑逐值复现）。

**结论与下一步**：真实双讲弱标签 = **项目首个打开声学全目标门的干预**（11 轮来第一次 B1+B2+B3+C1+early 同时严格达标，经三模型链校准）。剩余两墙：e2e 层（B4/B5/C3，对 DTD 敏感度的单调交易轴仍未破）+ A1 的 10 个 case 翻转（icassp2022 主导）。→ **R12 = 该轴剂量升级（真实双讲 mixture 15%→30%，v8r6）**：压测收益可扩展性/链余量（B1 余量 6%、B2 −36ms、C1 +1pt），并观察 e2e 是否随设备域学习自然改善。


## R12 详情（2026-10-16，否决；真实双讲剂量-反应：15% 最优、30% 反噬）

**唯一变量**：真实双讲注入份额 15%→30%（v8r6 = v8r1 hardlink 克隆 + 7,156 替换；标签/校验流程同 R11，p50 diff 0.0015）。

**训练/选点**：早停 ep195（patience 30），best valAP 0.9639（ep165）；anchor_nearest=e181（双轴优）、e165。

**standalone（双双反噬）**：
| | R11 e181（15%） | R12 e181（30%） | R12 e165（30%） |
|---|---|---|---|
| recall / rL95 | .9696/.9497 | .9653/.9444 | **.9718**/.9524 |
| FP/h | 42.58 | **91.85** | **97.19** |
| P95 | 1892 | 2583 | 2410 |
| early | 0 | **3** | **2** |

**链同参数对照（隔离剂量效应，s=0.8、veto=e157）**：
| | R11 链 lo=0.20（15%） | R12 链 lo=0.20（30%） |
|---|---|---|
| recall / rL95 | .9501/.92625 ✓ | .9479/.92371（余量变薄） |
| FP/h 点/U95 | 7.406/9.996 ✓ | **7.157/9.707**（更深） |
| **P95** | **1937 ✓** | **2144 ✗** |
| **early** | **0 ✓** | **1 ✗** |

lo=0.30 档同型（P95 2144/early 1/C1 余量仅 0.25pt）。**R12 链丢失 B2 与 early——无法维持声学全过**。

**结论**：30% 剂量复现 R8 型"剂量反噬"（standalone FP 翻倍、P95/early 病理化），融合端只能救回部分 FP，救不回 P95/early。**真实双讲轴的最优剂量=15%（R11）**。R11 链保持唯一全过配置；剩余目标 = e2e 层（B4/B5/C3）+ A1 十翻转。→ **R13 = 多增量链工具扩展 + 融合空间探索（免训练快轮）**：把 R6 系（低成本翻转 e144）与 R11 系（声学全过 e181）的增量叠进同一链，探索 A1 翻转减少 + e2e 改善的组合。


## R13 详情（2026-10-16，否决；融合探索收敛，R11 链确认为唯一最优）

**目的**：把 R6 系增量（e144，历史最少翻转=4）与 R11 系增量（e181，声学全过）叠进同一链，探索能否同时改善 A1 翻转与 FP 余量。**免训练轮**。

**工具扩展**：`export_fused_union_onnx.py` 增 `--increment2-checkpoint/--scale2`（任意 N 增量 + veto，state 按各模型层数堆叠 [13,1,288]）；`export_fused_veto_onnx.verify_parity` 的 state 构造泛化（原假设 2-3 模型）。

**回归自证**：用扩展工具重导 R11 链（lo020）与原 ONNX 对拍 **max|Δ| = 0.0（30 帧）**——扩展零行为漂移。

**四模型链（e147 + 0.6·e144 + 0.8·e181 + veto(e157, lo0.20)）**：
| | R11 链（3 模型） | R13 链（4 模型） |
|---|---|---|
| recall / rL95 | .9501/.92625 | .9501/.926250（**逐位相同**） |
| FP/h 点/U95 | 7.4063/9.9956 | 7.6556/10.2830（+0.25 劣） |
| P95 / early / noisy | 1937 / 0 / 28/30 | 1937 / 0 / 28/30（相同） |
| case 翻转（比对文件口径） | **13**（10 FP + 3 FN） | **14**（11 FP + 3 FN） |

**结论**：e144 增量在本融合族零贡献纯加劣（+0.25 FP、+1 翻转）——R6 系的低翻转性质来自其配对 veto（R2）与尺度 s=0.6 的组合，不可移植。**融合探索收敛：R11 链为唯一最优配置。** 顺带修正 R11 台账的翻转计数（判定器 severe 列表截断为 10，比对文件全量口径为 13：10 FP + 3 FN）。

## R14 详情（2026-10-16，契约扩展：v1 事故与修复，R14b 重训中）

**目标**：剩余两墙（A1 的 icassp2022 FP 翻转族 + e2e B4/B5/C3）的共因假设——DTD 只有 mic（AEC 前）+ render，缺"回波实际被消除到何种程度"的直接观测；契约扩展第三通道 = **post-AEC（CaptureAfterAec）**，echo-only 帧 AEC 输出静默、doubletalk 帧残留近端，可把"知道 echo 但判不了"的盲区拆开。

**用户批准（2026-10-16）**：ONNX 输入契约扩展获批，**且不限方向**（任意扩展方向均许可）。同时新增约束：**禁止多模型融合**——达标候选必须单模型推理（R11 链等仅作诊断证据）。

**全链路改动（四处）**：
- 训练侧：`dtd_model.py` `input_channels 2→3`（前端追加 aec 对数功率 + 压制比特征，feature 5→7/band）；`enriched_dataset.py` 谱域 AEC 代理通道（`spectral_aec_residual`，EMA 滤波估计）；`train_dtd.py --aec-input-channel`；`DTD_INCLUDE_AEC` 环境变量（Windows spawn worker 不继承模块属性，须走环境变量）。
- 导出：`export_dtd_onnx.py` 4 输入（`aec_output`）+ parity 校验。
- C#：`IRenderAwareDtdService` 4-span Process；`RenderAwareDtdOnnxRunner` 3/4 输入自适应（`HasAecOutputInput`）；`RenderAwareDtdService` 第三 overlap buffer；调用点喂 `CaptureAfterAecValid`（同帧可得）。
- **legacy 安全**：3 输入模型（e147）在新 C# 上逐位不变——e147 声学重放 0 diffs 通过（修复前后各一次）。

**v1 事故（R14a，作废）**：评测全盘异常（FP 389/887 = 基线 12 倍、negativeExposureHours 2.89）。根因 = **benchmark APM 块长 160 样本（10ms）→ post-AEC 流相对 mic/render 固有延迟 160 样本**；检测器 push 时用 `TryRead(num)` 读"当下 hop"，永远缺尾 → aec 通道全程喂**零**（或错位数据），与训练代理失配。**修复两处**：(1) 检测器改读"最新完整覆盖 hop"（`newestStart = Start + Count − 320`，对齐 APM 处理延迟）；(2) 训练代理 `spectral_aec_residual` 同加 160 样本右移。**v1 权重作废，R14b 以修复后代码重训**（配置仅 output-dir 不同，r14b-v8r5-aec.json）。

**R14b 训练（2026-10-16 23:40 暂停 @ep156 → 23:52 恢复）**：230ep 预算，attempt 1，~130s/epoch，best selection **1.36795 @ep155**（valAP 0.96276；运行中 ep158+）。因用户重启电脑于 **ep156 完成点** 主动停止（guard+trainer 进程树已终止，last.pt/156 checkpoints 完好）；重启后重跑 guard 同参数自动进入 resume 模式（`Find-ResumeCheckpoint` → last.pt），ep156–158 恢复后 valAP/selection 与重启前同带宽（无恢复副作用）。早停监控 = `AP+0.10·F1+0.10·R@0.1%FPR+0.15·R@0.5%FPR+0.25·R@1%FPR`（patience 30/min-delta 0.0002）；ETA：早停最早 ~00:58（ep185），跑满 ~02:34。

**AEC 模块兼容语义表（2026-10-16 核查）**：4 输入契约机械层对任何 `IAcousticEchoCancellationService` 泛型兼容（ingress 用其 DelaySamples）；但模型效力取决于『训练假设通道滞后=160 样本 + 线性消除器残差语义』与实模块匹配：WebRtcApm=160 ✓ 精确（benchmark 固定用此）；DTLN=128（默认 OutputPrerollSamples=HopLength，可配 160 对齐，免重训可换）；DFSMN=640（frameLength，错位 480 显著）。换模块三步：训练代理延迟设为目标 DelaySamples →（理想）用其残差做代理 → 到该模块下重跑 benchmark。

**R14b 判决（2026-10-17 凌晨，否决：训练/推理通道域不匹配——R14 的真正根因）**：
- **训练**：早停 @ep205（patience 30 耗尽），best selection 1.37379 @ep175（valAP .96407）；选点 anchor_nearest=181（rL95 .7189≥anchor、fpU95 47.351 恰等）/early_stop_best=175/ap_best=182。
- **e181 评测（灾难复现）**：FP **480**/887（192.7/h）、recall .9696（与 R11 e181 同值巧合——正例全触发）、early 6、errorCount 1（控制循环取消，非 DTD）；导出契约 4 输入 + parity ✓（1.1e-07）。
- **诊断链（多轮，工具/证据全部留档 `r14/diag-aec/`）**：(1) 评测日志只记 FP/FN（0 TP 行≠异常）；(2) 环境变量门控探针实测运行时 aec 通道**非零**且覆盖率随帧量化在 [num−160, num] 抖动；(3) 波形互相关（RAW hop 转储 + concat 法）判定 buffer 内容语义 = **到达位置**（content(p)=resid(p−160)，与 mic 互相关峰一致落在 +320 移位）；(4) C# 概率轨迹 vs Python ONNX 回放逐点对拍：中段逐点吻合（±0.01，证明回放忠实），尾段发散根因 = **运行中 ResetModelForPathChange**（case5 @51520，延迟跟踪器抖动触发）；(5) **决定性实验**：读 v8r5 验证集 ground truth——训练代理 far_only 通道 = **0.83×mic**（几乎不消回声），而运行时真 WebRtcApm = **0.08–0.6×mic**（强 12–20dB 压制）→ **训练/评测通道分布不相交**；模型在代理内容（验证集 valAP .964）正常、在真 APM 内容下狂暴（Python 直喂真通道：far-only case 概率底抬 0.4-0.5 + reset 爆发 0.71 ⇒ 假触发；喂代理/mic 通道则低沉）。
- **结论**：R14 前提（用真实 AEC 信息）成立，但**离线谱代理（lam/alpha/EMA、无延迟补偿）的消除强度与真实 AEC3 差 10 倍以上**——模型学到的是代理特有的退化依赖，部署即 OOD。离线代理路线判死。**R14c = 训练通道改为真实 APM 离线残差**：新工具 `src/Tools/VoiceBox.DtdAecPass`（同一 WebRtcApmService、同 benchmark 配置 HPF+AEC3+NS off+AGC off、同 256 样本节拍、确定性），对 v8r5 materialized train+val 全量生成 `_aec.wav` sidecar；loader 增 `DTD_AEC_SOURCE=apm` 加载 sidecar（前 160 样本=0 与旧代理约定一致，服务预填块延迟）；其余配方不变重训。
- **C# 读路径修复（保留）**：aec hop 读取改为**内容精确**语义——`TryRead(num)`（到达位置语义下该内容=残差[mic−160, mic+160)，与训练通道窗口精确对齐）；尾部未就绪时**推迟到下一 hop 边界**执行该 hop 的 DTD 调用（仅对 `HasAecOutputInput=true` 模型生效，3 输入 legacy 路径零变化）；接口增 `IRenderAwareDtdService.HasAecOutputInput` 默认成员。**探针已清除；e147 legacy 对照用最终版重跑 = 0 diffs（overall 0 / 58 切片 0 / 1349 case 决策 0，`r14/legacy-control-e147/`）。**

**R14c 结果（2026-10-17，否决，但确立了"通道一致性"铁律）**：sidecar real-APM 通道覆盖了 80%？— 实际发现四分支中仅 materialized(=wenet 20%) 分支走 sidecar，其余 80%（narrow/fullband/real_single_talk）仍用旧代理 → **混合通道训练**；评测 e185 FP 434（160.8/h）、e214 FP 475 → 混通道模型在 100% 真通道运行时依旧 OOD。**教训：任何通道方案必须在训练全分支与运行时两侧完全一致。**

**R14e/R14f 结果（2026-10-17，否决；机制打通 + 通道一致性假设全链路验证）**：
- **通道语义契约化**：ONNX metadata 新增 `aec_channel_source`（capture | spectral_proxy），`IRenderAwareDtdService.PostAecChannelSource` 默认成员，检测器据契约选择通道来源。
- **运行时谱代理（C# 移植）**：`SpectralAecProxy`（640/320/hann/λ1e-2/α0.6/延迟160，与 `enriched_dataset.spectral_aec_residual` 同语义）+ 恒定一 hop 延迟执行（复用 hold 机制）。**端口验证：单元自测（`SpectralAecProxySelfTest`，env 门控）vs numpy 参考 = corr 0.999979 / relMAE 0.038%**（修复一处 irfft 输入输出同址别名 bug；DFT/EMA/OLA 中间量逐级对拍均一致）。
- **评测（R14b e181 权重，proxy 契约重导出，proxy 运行时）**：**FP 480→64**（15.71/h vs 基线 7.903）、recall .9566/L95 .9339（基线 .9414/.9161 ↑）、P50 376、**P95 2037**、early **1** — smoke FP/h 32→全量 15.7 尺度一致。**判决：否决**（B1 FP 仍 2× 基线、B2/early 未过）；但对 R11 家系是实质改进（2-ch e181 FP/h 42.6 → 3-ch 15.7，recall 同档）。
- **机制结论**：4 输入契约 + 通道一致运行时可用且有效；FP 余量受 v8r5 家系高召回姿态限制。→ **R15 = 基线家系（v5 数据 + v6 冻结配方 onset 0.10）+ 同一第三通道**（proxy 通道在训练全分支天然一致）。

**R15 结案（2026-10-17，否决；通道方向测毕+交易轴终确认）**：基线家系（v5 数据 + v6-candidate-01 冻结配方 onset 0.10）+ 同一第三通道（proxy 语义，训练全分支一致）。230/230 全训（best_epoch 221，valAP .96727）；验证集 anchor_nearest pool=0（双轴被 anchor 支配，按先例明示偏离走 benchmark）。**ep221：TP 409/FN 52/FP 43（10.43/h）、recall .8872/L95 .8551、P50 398、P95 2148、early 0** → 通道在基线家系上双轴劣化（召回 −5.4pt + FP 仍 > 基线）。**两代基座 × 两代通道来源的四点印证同一"精度↔召回"单轴**：v5-2ch（.9414/33）、v5-3ch（.8872/43）、v8r5-2ch（.9696/~121）、v8r5-3ch（.9566/64）——第三通道仅在 v8r5 家系上产生净召回-换-FP 的位移，从未产生基线东北方向的点。**结论：通道扩展方向判耗尽**（与 R1–R3 的单轴结论闭环）。

**收敛后的战略状态（用户 2026-10-17 纠正："方向已穷举"结论过早）**：此前 15 轮主要在同一套模型上调数据配比/损失/大小/通道——不能证明单模型到极限。**用户重排方向优先级**：①基线约束微调（最推荐）②真实 AEC 全分支一致受控试验 ③跨时间回声相关特征 ④弱标签清理+定向难负样本 ⑤蒸馏+真标签纠偏（R16 纯模仿不够，验收看真标签成绩）。**不放宽达标标准；先①后②**。

**R16（蒸馏）按用户判断降级留档**：链蒸馏训练至 ~ep60 停止（checkpoint 保留，`r16/training/output/checkpoints`，可续）；设计结论 = 纯模仿教师会继承教师取舍，仅作为方向⑤基座。教师标签工具（`rounds/score_distill_teacher.py`，含 --fusion keep 基线与 chain 链两种模式，kind/val 49,500 条）保留复用。

**R17（方向①，2026-10-17 22:01 启动）**：**基线约束微调**——e147 权重初始化 + **前端冻结** + 小 LR（1e-4, 40ep, warmup 2, patience 15）+ v5 materialized 回放（wenet 权重 1.0）+ **锚定约束**（`--baseline-anchor-weight 0.5`：item[anchor_target] = e147 逐帧冻结概率，BCE 约束输出偏移，只允许标签强烈要求处纠错）。新增能力：train_dtd `--init-checkpoint/--freeze-frontend/--baseline-anchor-weight`（loss 记录 `loss_baseline_anchor`）；loader `DTD_ANCHOR_TARGETS`；锚定目标 49,500 条（keep-only 模式）。Smoke（1ep/64 items）通过。**部署=单模型 2ch，结构与推理成本不变。**

**R17 结案（2026-10-17，否决；但首次由纯单模型微调推动 P95）**：早停 @ep22（best ep7）。ep7：recall .9544/L95 .9314、FP 50（12.23/h）、P95 2144、early 1；**ep0（对照：e147 初值+1ep 微调）：recall .9566/L95 .9339、FP 45（10.94/h）、P95 1961、early 0**。ep0 姿态 = **P95 首次在非融合点严格低于基线（1961<1973，此前最好 e188 1969）** + 召回 +1.5pt + early 0；代价 FP 33→45（B1 ✗）+ A1 翻转（✗）。**单轴位移再现**（约束微调只能沿轴移动，未能东北向）——但"小步初始化式微调确实能推动 P95/shoulder"是新的机制发现。→ 方向②（真实 AEC 全分支）启动。

**R18（方向②，构建中 2026-10-17 23:5x）**：**真实 AEC 全分支一致受控试验**——新工具 `rounds/materialize_branches.py` 把三个流式分支（aec_narrow 9,078 / aec_fullband 8,988 / aec_real_single_talk 13,759）确定性物化为固定条目（**31,825/31,825 完成**，label sidecar = 原始 frame 标签，loader 支持 `_labels.npy`）；随后 `VoiceBox.DtdAecPass` 对全部分支（含既有 wenet 48k）生成真实 WebRtcApm 残差 sidecar；四个数据槽全部改为 materialized + `DTD_AEC_SOURCE=apm` → 训练/运行时 100% 真实 AEC 通道一致（4 输入模型，capture 契约 + 内容精确读+hold）。

**R18 结案（方向②，2026-10-18，否决；真实 AEC 全分支受控试验完成）**：训练早停 @ep186（patience 30 耗尽；best selection **ep156 = 1.326335**，valAP .93961），230ep 预算实际用 187ep。选点：policy 合格 0、anchor_nearest 池 0（R18 验证集 = 物化流式分支 + 真实 APM 通道，与 v5 anchor 分布不同）→ 按 R1/R15 先例**明示偏离**走 benchmark；唯一选点 **ep156**（early_stop_best）。**声学 e156：TP 437/FN 24/FP 91（23.16/h）、recall .9479/L95 .9237（+0.65pt）、P50 386、P95 2595、early 2、error 0**；B3 noisy/doubletalk_with_movement .9667/L95 **.8333（↑）**；FP 拆解 = icassp2022 **48** + test_set_noisy farend 19 + fullband_mobile 9 + 其它 15。**判定：B1/B2/early 明确失守（FP 2.9×、P95 +622ms、early 2）→ P7 veto，未消耗 e2e**。机制结论：**方向②的干净受控试验已完成——即使四分支与运行时 100% 使用同一真实 AEC 通道，模型仍落在同一"精度↔召回"单轴上（.9479/91 被基线 .9414/33 支配），未产生东北向位移**。五点测绘更新：v5-2ch(.9414/33) / v5-3ch-proxy(.8872/43) / v5-3ch-capture-real(.9479/91) / v8r5-2ch(.9696/~121) / v8r5-3ch(.9566/64)。**方向②结案**。→ 按用户优先级下一方向 = **③跨时间回声相关特征**。

## R19 详情（方向③，2026-10-18，否决；用户中途叫停并跑 benchmark 判定"无意义"）

**实现**：显式因果跨时间回声相关特征——前端在模型内部计算 per-band 窗口归一化 mic/render log-power 相关（500ms 窗口 W=25 帧，lags 0/1/2/4/8/16 帧 = 0–320ms），历史随 streaming state 携带（state 扩为 `[3+2*(25+16),1,288]=[85,1,288]`，ONNX 内自洽 → 训练/运行时天然一致，**C# 零改动**）。v5 数据 + v6 冻结配方、2ch（隔离特征轴）。冒烟：离线/流式逐帧等价 max|Δlogit|=9.2e-7；ONNX parity prob 8.9e-8 / state 1.9e-6；legacy e147 导出逐项不变（3 输入 / state 3,1,288 / 1,656,577 参数）。复杂度 **1,794,817 参数 / 99.64 MMAC/s（+7.4%）**。

**训练/停止**：06:07 启动，230ep 预算；用户要求 "停止训练，拿现有轮次数据跑 benchmark。如果有意义再继续" → **ep193（194/230）停止**，进程树清空。选点：policy 0 / anchor_nearest 0；评测 **ep171**（early_stop_best，sel 1.386625）与 **ep193**（ap_best，valAP .96653）。

**声学（vs 基线 .9414/.9161、FP 33/7.903、P50 396/P95 1973、early 0、B3 L95 .7032）**：

| | recall/L95 | FP (FP/h) | P50/P95 | early | B3 L95 |
|---|---|---|---|---|---|
| ep171 | .9046/.8743 | 53 (13.01) | 412/**1937** | 0 | **.7868** |
| ep193 | .9328/.9061 | 59 (14.56) | 412/2140 | 1 | **.7868** |

**判定：两点均被基线支配（召回 −0.9~−3.7pt 且 FP 1.7–1.8×）→ 不跑 e2e、不续训**。孤立亮点：ep171 **P95 1937（−36ms，非融合点历史最好之一）**、early 0、B3 L95 .7868（↑8.4pt）——但 A1/B1 不可过。FP 仍集中 icassp2022（31/35）+ noisy（11/13）；两点再次沿同一精度↔召回轴移动（ep171→ep193：召回 +2.8pt / FP +6 / P95 +203）。**机制：500ms 带功率相关是回声路径结构的冗余观测，未提供"mic 能量中哪部分不被 render 解释"的新信息** → 方向③（本实现形态）不构成东北向位移。→ 下一步候选：方向④（弱标签清理 + 定向难负样本）或 ③ 改型（在因果线性回声预测的**残差/创新**上做多延迟相关）。

## R20 详情（方向④前半，2026-10-18，否决；弱标签清理完成）

**关键测量（立项依据）**：R11 的 8,434 段 Wiener 残差弱标签，会话级回声路径拟合 **ERLE 中位仅 3.1dB（p90 11.1）** —— 大多数"近端残差"实为残留回声；R11 台账此前的"中位 +17.5dB"记录有误，已勘误。

**设计（单变量 vs R11）**：同剂量（15% mixture = 3,610 行），只注入 **ERLE ≥ 5dB** 的干净子集（3,269 段，四族齐全，movement 0.61，确定性重复保剂量）；其余与 R11 完全一致（onset 0.30、2ch、v8r1 基座）→ `v8r7`、config `r20-v8r7-clean-labels.json`。

**工具与数据事故（重要）**：`inject_real_doubletalk.py` 的 hardlink 克隆 + manifest 原地写导致 **v8r1/v8r5/v8r6/v8r7 的 manifest 同一 inode 被覆盖**（音频无损——wav 替换用 temp+replace）。恢复：新增 `prepare_materialized_v5.py --manifest-only` 确定性重生成 pristine v8r1 manifest（未注入的 **40,812 行逐字段一致**），新增 `rounds/replay_injection_manifest.py` 从 pristine 重放各轮注入——**v8r5 3,610 / v8r6 7,156 / v8r7 3,610 行与历史精确复现**，抽检 15/15 字节匹配。工具已修复：非 wav 不硬链、写必 temp+replace。教训入记忆 `dtd-hardlink-dataset-pitfall`。

**训练/选点**：早停 @ep222（best sel ep192 1.376642）；policy 0 / anchor_nearest 0；评测 ep192（early_stop_best）与 ep171（ap_best，valAP .96422）。

**声学（vs 基线 .9414/.9161、33/7.903、396/1973、early 0、B3 L95 .7032；vs R11 e181 .9696/42.58、378/1892）**：

| | recall/L95 | FP (FP/h) | P50/P95 | early | B3 L95 |
|---|---|---|---|---|---|
| ep192 | .9588/.9365 | 144 (37.98) | 364/2009 | 1 | .8333 |
| ep171 | .9610/.9391 | 175 (47.63) | 364/2410 | 2 | .8333 |

**判定：两点均被基线支配（B1 4.8–6×、B2/early 失守）→ 不跑 e2e**。与 R11 相比：**清洁标签只把同轴点轻移**（召回 −1.1pt 换 FP −4.6/h），无东北向位移。**FP 扩散到全部 farend 语料**（含基线为 0 的 device_real 10 / real_doubled 11、test_set_clean 16、noisy 45、icassp2022 49）——**弱标签污染不是 FP 暴涨的主因**，真实双讲材料让模型整体转向高敏感姿态。→ 方向④前半结案；后半（定向难负样本）仍待试。

## R21 详情（方向⑤首试，2026-10-18，否决；软概率蒸馏被证不适用）

**设计**：教师 = R11 链的逐帧软概率（`r16/distill-targets`，v5 数据），学生 = e147 初始化 + 冻结前端 + 真实标签主损失 + **0.5·教师软目标 BCE**，LR 2e-4、全 v5 mixture、80ep。

**训练**：早停 @ep20（best sel ep0 1.233684；锚定损失 0.577→0.445 收敛）；选点 policy 0/anchor 0；评测 ep0 与 ep20。

**声学（vs 基线 .9414/.9161、33/7.903、396/1973、early 0、B3 .7032）**：
| | recall/L95 | FP (FP/h) | P50/P95 | early |
|---|---|---|---|---|
| ep0 | **.7701**/.7295 | 21 (4.96) | 440/**2615** | 0 |
| ep20 | **.9414**/.9161（与基线逐 case 相同） | 38 (9.14) | 424/2172 | 0 |

**判定：否决**。**机制结论**：(1) 链的融合输出是**决策分数**而非校准概率，软目标 BCE 把学生整体压低（ep0 召回崩到 .77）；(2) 真实标签随后重新主导，ep20 精确漂回基线工作点（TP 434/FN 27 与基线完全一致）——**在 v5 域内蒸馏无法传递链的 OOD 决策**。→ ③④⑤ 均已有结论；⑤ 的修正实现见 R22。

**R22（方向⑤修正版，2026-10-18 21:16 启动）**：教师目标改为**链决策二值目标**（链融合分 ≥0.55，v8r7 帧上生成，正帧 34.86%），学生数据改为 **v8r7**（真实双讲清洁弱标签）——让学生看到真实双讲材料并学习链在该域上的 union+veto 决策（尤其"e181 想触发而 e157 否决"的 FP 抑制）。其余 = e147 初始化 + 冻结前端 + 真实标签主损失 + 0.5 权重 + LR 1e-4/40ep。

**R22 结案（2026-10-18，否决；但决策蒸馏是迄今最强的 veto 迁移机制）**：早停 @ep15（best sel ep0）；评测 ep0 与 ep15。**声学（vs 基线 .9414/7.903/1973）**：
| | recall/L95 | FP (FP/h) | P50/P95 | early | B3 L95 |
|---|---|---|---|---|---|
| ep0 | .8048/.7661 | 15 (3.51) | 504/2615 | 0 | .6644 |
| **ep15** | **.9566/.9339** | **58 (14.13)** | 416/2168 | 0 | .7032 |

**判定：否决**（B1 1.8×、B2 +195ms；ep0 早期锚点再次压制召回）。**机制收获**：**决策级蒸馏把 R11 家系的 FP 暴涨大幅压回**（R11 e181 42.58/h → R22 .9566/14.13/h，与 v8r5-3ch-proxy .9566/15.71 同档且无需第三通道）——veto 行为可迁移到单模型；但没有跨越同轴（相对基线仍是 +1.5pt 召回换 +6.2/h FP，P95 未改善）。ep0→ep15 说明锚点在早期主导、真实标签后期反超，两股力量在该配方下无法共存于 NE 方向。

**方向①–⑤ 全部完成至少一次受控试验**（① R17、② R18、③ R19、④ R20、⑤ R21+R22）。唯一声学全过配置仍是禁止的融合链；单模型前沿未被突破。

## R23 详情（方向⑤机制组合，2026-10-18，否决；迄今最强单模型点）

**设计**：从 R17 ep0（唯一非融合 P95 击穿点 .9566/45/1961）出发，用 R22 的链决策锚点在 v8r7 上短程微调（LR 5e-5、≤20ep、冻结前端）。早停于 ep8（selection 代理在 ep0 见顶）后**禁用 patience 并恢复训练至 ep20**（benchmark 姿态仍在恢复通道：召回 .809→.95、P95 2603→2172 单调改善）。

**声学轨迹（vs 基线 .9414/.9161、33/7.903、396/1973、early 0、B3 L95 .7032）**：
| | recall/L95 | FP (FP/h) | P50/P95 | early | B3 L95 |
|---|---|---|---|---|---|
| ep0 | .8091/.7707 | 17 (3.98) | 484/2603 | 0 | .6644 |
| ep8 | .9501/.9262 | 22 (5.18) | 422/2172 | 0 | .7032 |
| **ep12** | **.9544/.9314** | **29 (6.88)** | 416/2172 | **0** | .7032 |
| ep16/19 | .9523/.9288 | 29 (6.89) | 416/2172 | 0 | .7032 |

**A1 声学 strict compare（ep12，`-SkipEnvironmentStability` 明示偏离——dotnet-info 漂移与 R10 同型）**：**Regressed，105 findings = 7 个 acoustic-case + 98 acoustic-slice**。7 个 case 全部为**新增 FP**（device_real_doubled / clean / icassp2022×2 / noisy×3 movement），无新增 FN——历史最少 case 翻转之一（R11 链 13、R6 链 4）。

**判定：未达标**。**首个在聚合指标上同时严格优于基线 B1（6.88<7.903）与 C1（.9544>.9414）的单模型点，early 0，case 翻转仅 7**；剩余墙：**B2 P95 2172（ep8–19 钉死）**、**B3 L95 = 基线（无严格增益）**、A1 7 个新 FP case、A2 105 findings。e2e 未跑（门 veto；B4/B5 诊断可另行索取）。

**机制总结（23 轮）**：单模型已可达到"聚合 B1+C1 双优"，但（1）P95 尾部由固定 4-case 材料钉死，纯 DTD 干预只能在 shoulder 上微移；（2）A1 的零 case 回退要求 FP/FN 集合 case-subset，任何独立模型的 decision 边界都会引入新 FP；（3）noisy-move L95 基线 .7032 在 R23 中被精确复现而非提升。→ 下一步候选：在 R23 配方上做**B2/B3 定向干预**（onset/难负样本权重 + 更细的锚权重/学习率格点），或接受 A1 墙与用户讨论判定口径。

## R24–R26 详情（B2/B3 定向干预，2026-10-19，全部否决；单模型轴完全闭合）

**R24（R23 + onset 0.30）**：onset 在 e147-init 微调 regime 只把 P95 2172→2144、P50 416→402，**B3 钉死 .7032**，FP 随训练漂移 6.88→8.84。ep8 最佳（.9523/6.88/2144）。→ onset 的 B3/P95 收益只在从零长训练出现。

**R25（v8r7 从零 230ep + onset 0.30 + 决策锚 0.5，早停 ep43）轨迹映射**：
| ep | recall | FP/h | P50/P95 | early | B3 L95 |
|---|---|---|---|---|---|
| 13 | .9371 | 9.13 | 438/删失 | 2 | .6644 |
| 20 | .9328 | **7.16** | 482/2607 | 0 | .6644 |
| **28** | **.9610** | 15.76 | **374**/2222 | 0 | **.7438** |
| 36 | .9675 | 56.3 | 396/3232 | 3 | .8333 |
| 42 | .9631 | 28.5 | 428/2396 | 0 | .7438 |

**B3 首次严格改善（ep28）时 FP 已 2× 基线**；ep20 保 B1 但 C1/B3 均失守 → **无轨迹点同时过 B1+B3**。

**R26（R25 ep28 出发点 + 锚权重 1.0 短程 12ep）**：ep6 = **.9610/13.05/P50 408/P95 2384/early 0/ B3 L95 .7868（28/30，追平 R5 纪录）**；ep11 .9588/14.35/B3 .7438。**锚权重加倍只把 FP 15.76→13.05（仍 1.65×），B3 反而再升 → 锚饱和，无法同时进入低 FP 区**。

**方向⑤在三种 regime（从零、e147-init 微调、锚权重斜坡）下全部完成**。**单模型墙已精确刻画**：
- R23 ep12（.9544/6.88/B3 .7032/P95 2172）——低 FP 端
- R26 ep6（.9610/13.05/B3 .7868/P95 2384）——B3 端
- **B1↔B3 是严格单轴；B2（P95）由固定 4-case 尾部钉死；A1 要求零 case 翻转，任何独立决策边界都会引入新 FP**
- 唯一声学全过配置仍是禁止的融合链（R11）。单模型在现有冻结判决栈 + 标准下未达达标。

**待用户决策的下一步**：(a) 接受当前证据链，讨论判定口径/A1 口径或产品档位；(b) 继续在 B1↔B3 轴上做更细的权重格点（预期只能得到轴上的点，不能越轴）；(c) 新的结构方向（超出已列五方向）。

**R23 ep12 完整 e2e + 机器判定（2026-10-19 03:35，补记）**：`acceptance.json` = **FAIL**，但为**迄今最强单模型验收画像**：
- **PASS**：A3 / **B1**（点 6.88 与 U95 9.38 双降）/ **B5**（e2e falseAccept 29.60→**14.80**，腰斩）/ **C1**（rL95 .9314）/ **C2**（heard recall .8506→.8851）
- **FAIL**：A1（Regressed，7 个全新增 FP case、0 新 FN）/ A2（severe 222）/ B2（P95 2172）/ B3（=.7032）/ **B4（heard P95 1475→2792）** / **C3（drain P95 1478→2794）**
- **五项目标失败项中解决 2 项**（B1、B5），且 C1/C2 提升；代价是 e2e 尾延迟大幅恶化（保守锚使决策爬坡变慢）。方向⑤的三 regime 证据 + 该验收画像构成当前最完整的单模型决策依据。

**R27（尾延迟定向放松，2026-10-19，否决）**：从 R23 ep12 出发、锚权重 0.5→0.2、LR 3e-5、8ep。**ep2 .9544/38(9.10/h)/P50 416/P95 2144/early 0/B3 .7032；ep5/7 同型（9.36/9.10）**。**锚权放松只换来 FP 上抬（6.88→9.10，B1 失守），声学 P95 仍 2144（无延迟回收）** → 锚权重轴只交换 FP/veto，不买延迟；e2e 未跑。
**至此所有可及杠杆（数据、标签、特征、通道、蒸馏、锚权重、阈值）都落在同一精度↔召回/延迟轴上；唯一全过配置仍是禁止的融合链。** 决策包：`render-aware-dtd-decision-pack-20261019.md`。

**R28（方向④后半：定向难负样本，2026-10-19，否决但刷新最佳候选画像）**：从 R23 ep12 出发，hard-negative loss 0.15→0.40 + hard far-only 流概率 0.5→0.8（验证流不变），锚 0.5、LR 5e-5、10ep。
- 声学 ep5/9：**recall .9501/.9262、FP 24（5.67/h，U95 7.97）、P50 450–476、P95 2172、early 0、B3 .7032**。**B1 新纪录（点与 U95 双深降）**，代价 −0.43pt 召回（相对 R23）与 B3 持平。
- 完整 e2e+compare+判定（ep9，`-SkipEnvironmentStability`）：**FAIL，PASS A3/B1/B5/C1/C2**——B1 5.669/7.973、**B5 e2e falseAccept 9.87/h（基线 29.60，较 R23 的 14.80 再降 33%）**、C1 .9262、C2 .8621、early 0；FAIL A1（Regressed，**7 flips = 5 新 FP + 2 新 FN**）、A2（353）、B2 2172、B3 持平、B4 2798、C3 2801。
- **方向④两半（清理 + 难负样本）均已完成受控试验**。R28 ep9 与 R23 ep12 构成当前最佳两候选（前者 FP/e2e-FA 更深，后者召回/heard-recall 更高），同一组墙未变。
