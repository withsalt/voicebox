# wetext_processor_c native library

`wetext_processor_c` 是 [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) 1.3.0
`runtime/` 目录的 C++ 实现加上一层 C ABI，全部语言规则在 `.fst`
模型文件中，本库只负责：tagger FST 合成取最短路 → token 重排 → verbalizer FST 合成取最短路。

它服务于 `src/Thirdparty/WeTextProcessing.NET`（P/Invoke 封装），构建产物输出到该工程的
`runtimes/<rid>/native/` 目录。

## 与上游 1.3.0 runtime 的差异（均以 `[patch]` 注释标记）

| 文件 | 修改 | 原因 |
| --- | --- | --- |
| `wetext_processor.cc` | FST 加载失败 / 未知前缀抛异常（原为 `LOG(FATAL)`，会 `abort()` 整个宿主进程） | 库失败必须可恢复 |
| `wetext_processor.cc` | `Normalize` 取最短路前进行输出投影和 epsilon 消除 | 对齐 Python 1.3.0 默认归一化的等权候选选择；`Tag` / `Verbalize` 保留各自语义 |
| `wetext_token_parser.h` | 按 Python 1.3.0 保留规范顺序表之外的字段，拒绝重复字段 | 新版模型会输出 `value` 等额外字段，丢弃后无法 verbalize |
| `wetext_token_parser.cc/.h` | 按 Python 1.3.0 解码、重新转义字段值；未知解析类型抛异常 | 与 Python token 格式对齐，并允许宿主恢复 |
| `wetext_string.cc` | 非法 UTF-8 抛异常（原为 `CHECK_LE`） | 同上 |
| `wetext_processor_c_api.cc` | tag/verbalize/normalize 增加 try/catch，异常→返回 `nullptr` | 异常不能跨 C ABI |

1.3.0 上游 C++ 已包含英文 ITN、`preserve_order` 和严格的 token 分隔符检查。
本仓库保留可恢复异常与 C ABI 补丁，并同步 Python 新增的字段和转义语义。

随附模型覆盖中文/英文 TN、ITN，由官方 1.3.0 源码按默认参数编译；模型来源、生成工具版本和
SHA256 记录在模型目录的 `manifest.json`。金标准语料逐条比较 .NET 与 Python 输出。
.NET 的 `Normalize` 使用单最佳 tag → verbalize 流程，并同步 Python 1.3.0 的唯一输出路径选择。
Python 的多候选搜索（n-best）和文本位置映射 API 不在此封装范围内。

## 构建

依赖（OpenFst/gflags/glog）优先使用 `third_party/` 内的本地缓存包，否则由 CMake
FetchContent 从 GitHub 下载（需要网络），两种来源均做 SHA256 校验。无需手工准备任何预编译包。

两个一键脚本编译完成后都会**自动复制产物**到
`../../src/Thirdparty/WeTextProcessing.NET/runtimes/<rid>/native/`（需要提交进仓库）。

### Windows（win-x64）

需要 Visual Studio 2022 或更新版本，并安装“使用 C++ 的桌面开发”工作负载；CMake 3.14+ 与
Ninja 用 VS 自带的即可（脚本会自动定位）。在 `native/wetext-processor/` 用任意 PowerShell 执行：

```powershell
./build-windows.ps1
./build-windows.ps1 -Rid win-x64 -Config Release   # 可显式指定 RID / 配置
```

脚本会：定位 cmake/ninja（PATH 优先，否则用 VS 自带）→ 若 `cl.exe` 不在 PATH 自动导入
vcvars → CMake 配置/编译（Ninja 优先，缺失则回退 VS 生成器）→ 复制产物：

- `src/Thirdparty/WeTextProcessing.NET/runtimes/win-x64/native/wetext_processor_c.dll`
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

- `src/Thirdparty/WeTextProcessing.NET/runtimes/linux-x64/native/libwetext_processor_c.so`
- `src/Thirdparty/WeTextProcessing.NET/runtimes/linux-arm64/native/libwetext_processor_c.so`

## 升级上游

1. 以新版 pip 包 `runtime/` 目录为基线，重新套用上表补丁（搜 `[patch]`）；
2. 核对 `tn/token_parser.py` 和各语言 normalizer 中的 `TOKEN_ORDERS` 及转义语义；
3. 1.3.0 的 pip 包不再携带预编译 FST。使用测试目录的 `generate_golden.py`，从官方源码以
   `cache_dir=False` 构建并导出四组 FST，同时更新金标准及模型清单（见
   [生成说明](../../src/Thirdparty/Test/WeTextProcessing.Tests/TestData/README.md)）；
4. 重建并提交三个 RID 的原生库，再运行 `WeTextProcessing.Tests` 及依赖它的项目构建。
