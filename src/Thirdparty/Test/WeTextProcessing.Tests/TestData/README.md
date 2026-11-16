# Golden corpus

`*.tsv` 每行为 `输入\t期望输出`，期望输出由官方 Python WeTextProcessing **1.3.0**
的 `normalize` 生成，输入取自该版本的官方测试语料。`GoldenCorpusTests` 逐行断言
.NET 输出与 Python 一致，共 925 条。

1.3.0 不再随 pip 包发布预编译 FST。脚本从源码以默认参数、`cache_dir=False` 构建
中文/英文 TN、ITN，导出八个 FST 到 `src/Thirdparty/WeTextProcessing.NET/runtimes/models/WeTextProcessing`，
并使用同一组模型生成语料和包含 SHA256 的 `manifest.json`。任何生成失败都会中止，不跳过输入。

准备 Python 环境（本次使用 Python 3.12、conda-forge pynini 2.1.7 / OpenFst 1.8.4）：

```bash
conda create -n wetext-build -c conda-forge python=3.12 pynini=2.1.7 importlib_resources
```

下载并解压 [官方源码包](https://pypi.org/project/WeTextProcessing/1.3.0/#files)，
核对 SHA256 为 `217a939201fc1e3324c67bc9983c4f4e56dc9dc37b850845b3d2e0a1daf6fb5e`。
从本目录执行（示例为 Windows 路径）：

```bash
conda run --no-capture-output -n wetext-build python generate_golden.py E:\Temp\wetextprocessing-1.3.0
```

| 文件 | 方向 | 条数 |
| --- | --- | --- |
| zh_tn.tsv | 中文 TN | 137 |
| zh_itn.tsv | 中文 ITN | 206 |
| en_tn.tsv | 英文 TN | 114 |
| en_itn.tsv | 英文 ITN | 468 |

Python 1.3.0 的 `normalize` 在取最短路前进行输出投影和 epsilon 消除；
等权候选的选择可能与 `verbalize(tag(text))` 不同（如英文 `1000`、`1001`、`2000`）。
原生封装同步此选择方式，金标准不接受任意替代读法。
