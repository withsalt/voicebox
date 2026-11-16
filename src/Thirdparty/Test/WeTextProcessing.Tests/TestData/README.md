# Golden corpus

`*.tsv` 每行为 `输入\t期望输出`，期望输出由 Python WeTextProcessing 包（与
`runtimes/models/WeTextProcessing` 中相同版本的预编译 FST）生成，输入取自该包的官方
测试语料。`GoldenCorpusTests` 逐行断言 .NET 输出与 Python 完全一致。

升级 FST 模型后重新生成（需要 pynini，可用 conda `voicebox` 环境）：

```bash
python generate_golden.py E:\Temp\wetextprocessing-1.2.0
```

| 文件 | 方向 | 条数 |
| --- | --- | --- |
| zh_tn.tsv | 中文 TN | 137 |
| zh_itn.tsv | 中文 ITN | 206 |
| en_tn.tsv | 英文 TN | 111 |
| en_itn.tsv | 英文 ITN | 468 |
