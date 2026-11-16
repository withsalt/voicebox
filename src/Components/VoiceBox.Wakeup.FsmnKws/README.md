# FSMN KWS 推理资源

部署只需 `kws_streaming.onnx` 和同级 `kws_meta.json`。服务从 meta 读取关键词、token id 和 CMVN，
不读取 `am.mvn`、`tokens.txt` 或 `lexicon.txt`。

`cmvn.add_shift` 和 `cmvn.rescale` 必须各包含 400 个有限数值，rescale 必须为正。
缺失、维度错误或数值无效会在加载时抛出 `InvalidDataException`，不提供文件回退。
`CmvnFilePath` 配置已移除，调用方应删除此配置项。

`keywords.json` 不会自动加载；使用 `FsmnKwsOptions.Threshold` 和 `MaxKeywordSeconds` 等选项配置检测器。
