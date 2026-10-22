# third_party

依赖源码包的本地缓存。存在时 CMake 优先使用（见 `../cmake/*.cmake`），否则回退到
GitHub 下载；两种来源均做 SHA256 校验，内容完全等价。

| 文件 | 上游 |
| --- | --- |
| gflags-2.3.0.tar.gz | https://github.com/gflags/gflags/archive/refs/tags/v2.3.0.tar.gz |
| glog-0.7.1.zip | https://github.com/google/glog/archive/refs/tags/v0.7.1.zip |
| openfst-1.8.5-2026-04-11.tar.gz | https://github.com/csukuangfj/openfst/archive/refs/tags/v1.8.5-2026-04-11.tar.gz |
