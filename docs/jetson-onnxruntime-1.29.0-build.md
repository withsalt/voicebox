# Jetson 上编译 ONNX Runtime 1.29.0

本文说明如何在以下环境中原生编译 ONNX Runtime Python wheel：

- NVIDIA Jetson，8 GB 统一内存。
- JetPack 7.2 / Jetson Linux R39.2。
- CUDA 13.2。
- cuDNN 9.x。
- ONNX Runtime 1.29.0。

建议先只构建 CUDA Execution Provider。确认 CUDA EP 可以正常加载和推理后，再按需构建
TensorRT EP。

## 兼容性说明

ONNX Runtime 1.29.x 官方 GPU 包对应 CUDA 13.x 和 cuDNN 9.x。CUDA 13.0 构建产物可以在
CUDA 13.x 同一主版本内运行，因此 CUDA 13.2 在支持范围内。

ONNX Runtime 1.29.0 要求 CMake 3.28 或更高版本。JetPack 7.2 默认使用 CUDA 13.2；系统应显示
Jetson Linux R39.2。

Jetson 设备的 GPU 架构并不都相同。编译时优先使用：

```text
CMAKE_CUDA_ARCHITECTURES=native
```

这样只生成当前设备需要的 CUDA 代码，可以显著降低编译时间、内存和磁盘占用。只有确认设备为
Jetson Orin 时，才可以显式改成 `87-real`。

## 1. 检查系统环境

在 Jetson 终端中执行：

```bash
cat /etc/nv_tegra_release
uname -m
readlink -f /usr/local/cuda
/usr/local/cuda-13.2/bin/nvcc --version
gcc --version
cmake --version
python3 --version
```

预期结果：

- `/etc/nv_tegra_release` 显示 `R39`、`REVISION: 2`。
- `uname -m` 显示 `aarch64`。
- `nvcc` 显示 CUDA 13.2。
- CMake 不低于 3.28。
- cuDNN 为 9.x。

检查 cuDNN 头文件和动态库：

```bash
find /usr/include /usr/local/cuda-13.2/include \
  -name cudnn.h -print 2>/dev/null

ldconfig -p | grep -E 'libcudnn.so|libcublas.so'
```

如果系统不是 R39.2，或 CUDA/cuDNN 开发组件不完整，应先修复 JetPack 安装，再编译 ONNX
Runtime。不要把 CUDA 12 或 cuDNN 8 的路径混入当前环境。

## 2. 安装编译依赖

```bash
sudo apt update

sudo apt install -y --no-install-recommends \
  build-essential \
  git \
  cmake \
  ninja-build \
  pkg-config \
  python3-dev \
  python3-venv \
  python3-pip \
  libopenblas-dev \
  zlib1g-dev
```

如果找不到 `cudnn.h` 或 `libcudnn.so.9`，检查当前软件源提供的 cuDNN 9 开发包：

```bash
apt-cache policy libcudnn9-dev-cuda-13
```

包可用时安装：

```bash
sudo apt install -y libcudnn9-dev-cuda-13
```

也可以通过 JetPack 元包补齐 CUDA、cuDNN 和 TensorRT 开发组件：

```bash
sudo apt install -y nvidia-jetpack
```

如果系统 CMake 低于 3.28，可以稍后在 Python 虚拟环境中安装新版 CMake：

```bash
python -m pip install 'cmake>=3.28,<4'
hash -r
cmake --version
```

## 3. 准备磁盘和 Swap

源码、依赖和中间构建文件会占用较多空间。开始前建议至少保留 35 GB 可用磁盘空间：

```bash
df -h /
free -h
swapon --show
```

8 GB 内存容易在 NVCC 编译阶段触发 OOM。建议临时增加 12–16 GB Swap，最好放在 NVMe，而
不是长期放在 microSD。以下示例创建 16 GB Swap：

```bash
sudo fallocate -l 16G /swapfile-ort
sudo chmod 600 /swapfile-ort
sudo mkswap /swapfile-ort
sudo swapon /swapfile-ort

free -h
swapon --show
```

此处不修改 `/etc/fstab`，因此 Swap 不会自动永久启用。编译和验证结束后，可以删除：

```bash
sudo swapoff /swapfile-ort
sudo rm /swapfile-ort
```

## 4. 下载 ONNX Runtime 1.29.0

使用普通用户执行以下命令，不要使用 `sudo ./build.sh`：

```bash
mkdir -p "$HOME/src" "$HOME/venvs"
cd "$HOME/src"

git clone \
  --branch v1.29.0 \
  --depth 1 \
  --recursive \
  --shallow-submodules \
  https://github.com/microsoft/onnxruntime.git \
  onnxruntime-1.29.0

cd onnxruntime-1.29.0
git describe --tags --exact-match
```

最后一条命令应输出：

```text
v1.29.0
```

## 5. 创建 Python 虚拟环境

```bash
python3 -m venv "$HOME/venvs/ort-1.29"
source "$HOME/venvs/ort-1.29/bin/activate"

python -m pip install --upgrade \
  pip setuptools wheel packaging numpy
```

后续编译、安装和验证都应保持该虚拟环境处于激活状态。这样不会和系统自带的 Python 包冲突。

## 6. 配置 CUDA 和 cuDNN 路径

Jetson 上 cuDNN 的库通常位于 `/usr/lib/aarch64-linux-gnu`，头文件通常位于 `/usr/include`。

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export CUDNN_HOME=/usr/lib/aarch64-linux-gnu
export CUDNN_PATH=/usr

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -x "$CUDA_HOME/bin/nvcc"
```

如果 `/usr/local/cuda` 已正确指向 CUDA 13.2，也可以将 `CUDA_HOME` 设置为 `/usr/local/cuda`。

### CUDA 13.2 的 CCCL/GCC 13 兼容处理

JetPack 7.2 当前提供的 `cuda-cccl-13-2 13.2.86-1` 包含一处无效的 C++ 模板特化语法；
使用 GCC 13 编译时会在 `cub/device/device_transform.cuh:44` 报错。NVIDIA 已在 CCCL
[提交 `8e41eea`](https://github.com/NVIDIA/cccl/commit/8e41eeabe54ab9ae48ad5640cfa7153b0a1071af)
中修复该问题。不要直接修改 `/usr/local/cuda`，可以建立仅包含两个头文件的覆盖目录：

```bash
CCCL_SYSTEM="$CUDA_HOME/targets/sbsa-linux/include/cccl"
CCCL_OVERLAY="$HOME/src/cccl-13.2.86-gcc13-overlay"

if grep -q 'struct ::cuda::proclaims_copyable_arguments' \
  "$CCCL_SYSTEM/cub/device/device_transform.cuh"; then
  mkdir -p "$CCCL_OVERLAY"

  (
    cd "$CCCL_SYSTEM"
    cp --parents \
      cub/device/device_transform.cuh \
      cub/device/dispatch/tuning/tuning_transform.cuh \
      "$CCCL_OVERLAY"
  )

  curl -fsSL \
    https://github.com/NVIDIA/cccl/commit/8e41eeabe54ab9ae48ad5640cfa7153b0a1071af.patch \
    | patch -d "$CCCL_OVERLAY" -p2 --forward --no-backup-if-mismatch
fi

export NVCC_PREPEND_FLAGS="-I$CCCL_OVERLAY"
```

`NVCC_PREPEND_FLAGS` 必须在运行 `build.sh` 的同一个 Shell 中保持有效。该覆盖目录只修正 CCCL
头文件的语法，不会替换 CUDA 库，也不会改变系统安装。

## 7. 编译 CUDA EP Python wheel

以下命令针对 8 GB 内存环境，以构建稳定性为优先：

```bash
./build.sh \
  --config Release \
  --build_dir build-jetson-cuda \
  --update \
  --build \
  --build_wheel \
  --use_cuda \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --cmake_generator Ninja \
  --parallel 1 \
  --nvcc_threads 1 \
  --flash_nvcc_threads 1 \
  --skip_tests \
  --cmake_extra_defines \
     CMAKE_CUDA_ARCHITECTURES=native \
     onnxruntime_BUILD_UNIT_TESTS=OFF \
     onnxruntime_USE_FLASH_ATTENTION=OFF \
     onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=ON
```

关键参数：

| 参数 | 作用 |
| --- | --- |
| `CMAKE_CUDA_ARCHITECTURES=native` | 只为本机 GPU 生成代码，避免构建多套 CUDA 架构。 |
| `--parallel 1` | 同时只执行一个编译任务，降低峰值内存。 |
| `--nvcc_threads 1` | 每个 CUDA 编译任务只使用一个 NVCC 线程。 |
| `--flash_nvcc_threads 1` | 限制 Flash Attention 编译线程。 |
| `--skip_tests` | 不构建和运行完整测试集。 |
| `onnxruntime_BUILD_UNIT_TESTS=OFF` | 在 CMake 层关闭单元测试目标。 |
| Flash Attention 为 `OFF` | 避免编译最重的一组 Flash Attention CUDA 内核。 |
| Memory Efficient Attention 为 `ON` | 满足 ONNX Runtime 1.29.0 当前 CUDA 源码的符号依赖。 |

如果设备明确为 Jetson Orin Nano、Orin NX 或 AGX Orin，可以把：

```text
CMAKE_CUDA_ARCHITECTURES=native
```

替换为：

```text
CMAKE_CUDA_ARCHITECTURES=87-real
```

第一轮编译不建议增加并行度。确认构建稳定后，如果设备持续有足够空闲内存，可以尝试
`--parallel 2`，但仍应保留 `--nvcc_threads 1`。

### Attention 配置说明

在 ONNX Runtime 1.29.0 中，不要同时关闭 Flash Attention 和 Memory Efficient Attention。两者同时
为 `OFF` 时，`onnxruntime/core/providers/cuda/llm/attention.cc` 可能因为找不到
`kCutlassSafeMaskFilterValue` 而编译失败。8 GB Jetson 上已验证的组合是：

```text
onnxruntime_USE_FLASH_ATTENTION=OFF
onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=ON
```

如果确实需要 Flash Attention，可把第一项改为 `ON`。修改任一 CMake 选项后应重新执行带 `--update`
的完整配置命令；配置差异较大时最好使用新的 `--build_dir`。

## 8. 安装 wheel

构建产物位于：

```text
build-jetson-cuda/Release/dist/
```

检查并安装：

```bash
ls -lh build-jetson-cuda/Release/dist/

python -m pip install --force-reinstall \
  build-jetson-cuda/Release/dist/onnxruntime_gpu-1.29.0-*.whl
```

不要在同一个 Python 环境中同时安装 `onnxruntime` CPU 包和 `onnxruntime-gpu`。当前文档使用的是
新建虚拟环境，因此通常不需要额外卸载。

## 9. 验证 CUDA Execution Provider

先检查版本和可用 Provider：

```bash
python - <<'PY'
import onnxruntime as ort

print("ORT version:", ort.__version__)
print("ORT device:", ort.get_device())
print("Providers:", ort.get_available_providers())

assert ort.__version__ == "1.29.0"
assert "CUDAExecutionProvider" in ort.get_available_providers()
PY
```

再使用源码仓库自带的小模型创建 CUDA Session：

```bash
python - <<'PY'
import onnxruntime as ort

model = "onnxruntime/test/testdata/mul_1.onnx"
session = ort.InferenceSession(
    model,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print("Session providers:", session.get_providers())
assert session.get_providers()[0] == "CUDAExecutionProvider"
PY
```

如果第二段脚本成功，说明 CUDA Provider 动态库及其主要依赖已经能够加载。

## 10. 可选：构建 TensorRT EP

先确认 CUDA EP 构建和验证均成功，再检查 TensorRT 开发文件：

```bash
dpkg-query -W 'libnvinfer*' 2>/dev/null
find /usr/include -name NvInfer.h -print 2>/dev/null
ldconfig -p | grep -E 'libnvinfer|libnvonnxparser'
```

使用单独的构建目录编译 CUDA + TensorRT wheel：

```bash
./build.sh \
  --config Release \
  --build_dir build-jetson-tensorrt \
  --update \
  --build \
  --build_wheel \
  --use_cuda \
  --use_tensorrt \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --tensorrt_home /usr/lib/aarch64-linux-gnu \
  --cmake_generator Ninja \
  --parallel 1 \
  --nvcc_threads 1 \
  --flash_nvcc_threads 1 \
  --skip_tests \
  --cmake_extra_defines \
     CMAKE_CUDA_ARCHITECTURES=native \
     onnxruntime_BUILD_UNIT_TESTS=OFF \
     onnxruntime_USE_FLASH_ATTENTION=OFF \
     onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=ON
```

安装新的 wheel 后，验证 Provider：

```bash
python - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print(providers)

assert "TensorrtExecutionProvider" in providers
assert "CUDAExecutionProvider" in providers
PY
```

创建 Session 时建议按以下顺序保留回退路径：

```python
providers = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
```

TensorRT 在首次加载模型时可能需要构建 engine，这个过程也会使用大量内存。8 GB 设备应限制模型
输入尺寸和 TensorRT workspace，并启用 engine cache，避免每次启动都重新构建。

## 常见问题

### `cicc died due to signal 9`、`cc1plus: Killed` 或系统卡死

这是典型的内存不足：

1. 确认 Swap 已启用且有剩余空间。
2. 确认使用 `--parallel 1`。
3. 确认使用 `--nvcc_threads 1 --flash_nvcc_threads 1`。
4. 关闭桌面、浏览器和其他占用统一内存的进程。
5. 保持只编译 `CMAKE_CUDA_ARCHITECTURES=native`。

检查内核是否触发 OOM Killer：

```bash
dmesg -T | grep -Ei 'out of memory|oom|killed process' | tail -n 30
```

### `attention.cc` 找不到 `kCutlassSafeMaskFilterValue`

这通常表示 ONNX Runtime 1.29.0 被配置成同时关闭了 Flash Attention 和 Memory Efficient
Attention。检查实际的 CMake 缓存：

```bash
grep -E '^onnxruntime_USE_(FLASH|MEMORY_EFFICIENT)_ATTENTION:' \
  build-jetson-cuda/Release/CMakeCache.txt
```

将 `onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION` 改为 `ON`，然后重新执行第 7 节中包含 `--update`
的完整命令。仅运行 `--build` 不会改变已经缓存的 CMake 配置。

### `paged_attention.cc:297` 报 `unused variable 'flash_block_size_ok'`

这是 Flash Attention 为 `OFF` 时触发的 ONNX Runtime 1.29.0 源码告警，而项目又使用了 `-Werror`。
在源码仓库根目录应用下面的一行兼容补丁：

```bash
file=onnxruntime/contrib_ops/cuda/bert/paged_attention.cc

sed -i \
  's/^  const bool flash_block_size_ok =/  [[maybe_unused]] const bool flash_block_size_ok =/' \
  "$file"

git diff --check -- "$file"
git diff -- "$file"
```

该属性只消除条件编译造成的未使用变量告警，不会改变运行逻辑。补丁后可使用相同构建目录继续
`--build`，无需删除已完成的对象。

### `device_transform.cuh:44` 报 `global qualification of class name is invalid`

这不是内存不足，而是 CUDA 13.2 随附 CCCL 头文件的已知无效 C++ 语法。执行第 6 节的 CCCL
覆盖步骤，并确认恢复编译前环境变量存在：

```bash
echo "$NVCC_PREPEND_FLAGS"

./build.sh \
  --build \
  --build_wheel \
  --config Release \
  --build_dir build-jetson-cuda \
  --parallel 1 \
  --use_cuda \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --skip_tests \
  --nvcc_threads 1
```

Ninja 会从第一个未完成目标继续，不会重新编译所有已成功对象。

### `Could NOT find CUDNN` 或找不到 `cudnn.h`

检查：

```bash
find /usr /usr/local/cuda-13.2 -name cudnn.h -print 2>/dev/null
ldconfig -p | grep libcudnn
echo "$CUDNN_HOME"
echo "$CUDNN_PATH"
```

确认安装的是 cuDNN 9 开发包，而不只是运行时包，并保留：

```bash
export CUDNN_HOME=/usr/lib/aarch64-linux-gnu
export CUDNN_PATH=/usr
```

### `libcudnn.so.9`、`libcublas.so` 或其他动态库找不到

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sudo ldconfig
ldconfig -p | grep -E 'libcudnn.so.9|libcublas.so'
```

如需检查 wheel 中 Provider 的缺失依赖：

```bash
ORT_CAPI_DIR="$(python - <<'PY'
from pathlib import Path
import onnxruntime

print(Path(onnxruntime.__file__).parent / "capi")
PY
)"

find "$ORT_CAPI_DIR" -maxdepth 1 -name '*.so' -print
ldd "$ORT_CAPI_DIR/libonnxruntime_providers_cuda.so" | grep 'not found' || true
```

### `no kernel image is available for execution on the device`

通常是 CUDA 架构编错。清理到新的构建目录并使用：

```text
CMAKE_CUDA_ARCHITECTURES=native
```

Orin 设备也可以显式使用 `87-real`，但不要把 Orin 的架构值套用到其他 Jetson 产品。

### `CUDAExecutionProvider` 没有出现在列表中

确认安装的是刚刚生成的 `onnxruntime_gpu` wheel，而不是 PyPI 的 CPU 包：

```bash
python -m pip show onnxruntime onnxruntime-gpu
python -m pip list | grep onnxruntime
python -c 'import onnxruntime as ort; print(ort.__file__); print(ort.get_available_providers())'
```

必要时在当前虚拟环境中卸载冲突包，再重新安装本地 wheel：

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install build-jetson-cuda/Release/dist/onnxruntime_gpu-1.29.0-*.whl
```

### CMake 或依赖下载失败后如何重试

网络错误可以直接重新执行原构建命令。若修改过 CUDA 架构、Provider 或其他 CMake 选项，应改用新的
`--build_dir`；不要在未知状态下混用旧缓存。

## 参考资料

- [ONNX Runtime CUDA Execution Provider 兼容性](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime Execution Provider 构建说明](https://onnxruntime.ai/docs/build/eps.html)
- [ONNX Runtime 1.29.0 Release Notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0)
- [ONNX Runtime 1.29.0 CMake 配置](https://github.com/microsoft/onnxruntime/blob/v1.29.0/cmake/CMakeLists.txt)
- [NVIDIA JetPack 7.2 设置说明](https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/latest/setup_jetpack.html)
- [NVIDIA CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html)
- [NVIDIA cuDNN Linux 安装说明](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)
- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [NVIDIA CCCL 无效模板特化语法修复](https://github.com/NVIDIA/cccl/commit/8e41eeabe54ab9ae48ad5640cfa7153b0a1071af)
