# wetext_processor_c native library

`wetext_processor_c` 是 [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) 1.2.0
`runtime/` 目录的 C++ 实现加上一层 C ABI，全部语言规则在 `.fst`
模型文件中，本库只负责：tagger FST 合成取最短路 → token 重排 → verbalizer FST 合成取最短路。

它服务于 `src/SDKs/WeTextProcessing.NET`（P/Invoke 封装），构建产物输出到该工程的
`runtimes/<rid>/native/` 目录。

## 与上游 1.2.0 runtime 的差异（均以 `[patch]` 注释标记）

| 文件 | 修改 | 原因 |
| --- | --- | --- |
| `wetext_processor.cc` | 识别 `en_itn_` 前缀 → `kEN_ITN` | 上游 runtime 不支持英文 ITN，加载即崩溃 |
| `wetext_processor.cc` | FST 加载失败 / 未知前缀抛异常（原为 `LOG(FATAL)`，会 `abort()` 整个宿主进程） | 库失败必须可恢复 |
| `wetext_token_parser.cc` | `ITN_ORDERS` 与 Python 1.2.0 `tn/token_parser.py` 完全同步（含 `preserve_order`/`units`/`quantity`/`zone`/`telephone`/`electronic`），zh/en ITN 共用同一张表 | 上游 C++ 表落后于 Python，英文 ITN 会丢字段 |
| `wetext_token_parser.h` | `Token::String` 支持 `preserve_order: "true"` 保留 tagger 输出顺序 | 与 Python 行为对齐，英文 ITN 日期依赖此语义 |
| `wetext_token_parser.cc` | 解析失败抛异常（原为 glog `CHECK_*`，会 `abort()`）；未闭合引号值原版死循环，现抛异常 | 对外公开的 `Verbalize` 可能收到任意输入 |
| `wetext_string.cc` | 非法 UTF-8 抛异常（原为 `CHECK_LE`） | 同上 |
| `wetext_processor_c_api.cc` | tag/verbalize/normalize 增加 try/catch，异常→返回 `nullptr` | 异常不能跨 C ABI |

对**合法**输入（Python 实现不会报错的输入），上述修改不改变任何输出，与 Python 1.2.0 逐字节一致
（由 `WeTextProcessing.Tests` 的金标准语料回归保证）。

## 构建

依赖（OpenFst/gflags/glog）优先使用 `third_party/` 内的本地缓存包，否则由 CMake
FetchContent 从 GitHub 下载（需要网络），两种来源均做 SHA256 校验。无需手工准备任何预编译包。

两个一键脚本编译完成后都会**自动复制产物**到
`../../src/SDKs/WeTextProcessing.NET/runtimes/<rid>/native/`（需要提交进仓库）。

### Windows（win-x64）

需要 Visual Studio 2022 或更新版本，并安装“使用 C++ 的桌面开发”工作负载；CMake 3.14+ 与
Ninja 用 VS 自带的即可（脚本会自动定位）。在 `native/wetext-processor/` 用任意 PowerShell 执行：

```powershell
./build-windows.ps1
./build-windows.ps1 -Rid win-x64 -Config Release   # 可显式指定 RID / 配置
```

脚本会：定位 cmake/ninja（PATH 优先，否则用 VS 自带）→ 若 `cl.exe` 不在 PATH 自动导入
vcvars → CMake 配置/编译（Ninja 优先，缺失则回退 VS 生成器）→ 复制产物：

- `src/SDKs/WeTextProcessing.NET/runtimes/win-x64/native/wetext_processor_c.dll`
- 同目录下的 `wetext_processor_c.pdb`（若存在）

### Linux / WSL（linux-x64 和 linux-arm64）

先安装基础依赖（Ubuntu / Debian / WSL）：

```bash
sudo apt update
sudo apt install -y cmake g++ make ninja-build
```

编译 linux-x64（x86_64 主机原生编译）：

```bash
cd native/wetext-processor
chmod +x ./build-linux.sh
./build-linux.sh x64          # 也接受 ./build-linux.sh linux-x64
```

编译 linux-arm64：

- 在 aarch64 主机上：`./build-linux.sh arm64` 即为原生编译。
- 在 x86_64 主机（含 WSL）上交叉编译，需额外安装 aarch64 gcc 工具链：

```bash
sudo apt install -y g++-aarch64-linux-gnu
cd native/wetext-processor
./build-linux.sh arm64
```

脚本会做主机架构检测与工具链前置检查（缺包会给出 apt 安装提示），有 Ninja 则优先使用，
编译完成后复制产物并打印导出符号：

- `src/SDKs/WeTextProcessing.NET/runtimes/linux-x64/native/libwetext_processor_c.so`
- `src/SDKs/WeTextProcessing.NET/runtimes/linux-arm64/native/libwetext_processor_c.so`

## 升级上游

1. 以新版 pip 包 `runtime/` 目录为基线，重新套用上表补丁（搜 `[patch]`）；
2. 核对 `tn/token_parser.py` 的三张 orders 表是否变化；
3. 替换 `src/SDKs/WeTextProcessing.NET/runtimes/models/WeTextProcessing/*.fst` 为新版 pip 包内的预编译 FST；
4. 用 `WeTextProcessing.Tests` 重新生成金标准并跑回归（见测试工程 README）。
