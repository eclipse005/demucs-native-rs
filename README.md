# demucs-native-rs

Hand-written **CPU + CUDA** inference for **HTDemucs v4** (no [burn](https://github.com/tracel-ai/burn)).

| Platform | CPU | CUDA |
|----------|-----|------|
| Windows x64 | ✅ | ✅ (CUDA 12.x toolkit + driver) |
| Linux x64 | ✅ | ✅ (CUDA 12.x toolkit + driver) |
| macOS arm64 / x64 | ✅ | — |

Package: `demucs-core-native` · CLI binary: `demucs-native`

## Download prebuilt binaries

GitHub Releases ship platform archives (models are **not** included):

| Archive | Contents |
|---------|----------|
| `demucs-native-*-linux-x64-cpu.tar.gz` | CPU-only Linux |
| `demucs-native-*-linux-x64-cuda12.tar.gz` | Linux + CUDA 12 |
| `demucs-native-*-windows-x64-cpu.zip` | CPU-only Windows |
| `demucs-native-*-windows-x64-cuda12.zip` | Windows + CUDA 12 |
| `demucs-native-*-macos-arm64-cpu.tar.gz` | Apple Silicon CPU |

CUDA builds need a local **CUDA 12.x driver + runtime** (toolkit not required at runtime when using dynamic loading). They are compiled on GitHub runners **without a GPU**; real GPU testing is done on your machine. NVRTC compiles kernels for the **device arch at runtime** (not hard-coded to a specific SM).

### Linux

```bash
tar xzf demucs-native-v0.1.0-rc3-linux-x64-cuda12.tar.gz
cd demucs-native-v0.1.0-rc3-linux-x64-cuda12
# put *.safetensors in ./models (see below)
./demucs-native -i sample.wav -o ./out-cpu -m htdemucs --model-dir ./models --device cpu
./demucs-native -i sample.wav -o ./out-cuda -m htdemucs --model-dir ./models --device cuda
```

### Windows

```powershell
Expand-Archive demucs-native-v0.1.0-rc3-windows-x64-cuda12.zip
cd demucs-native-v0.1.0-rc3-windows-x64-cuda12
.\demucs-native.exe -i sample.wav -o .\out-cpu -m htdemucs --model-dir .\models --device cpu
.\demucs-native.exe -i sample.wav -o .\out-cuda -m htdemucs --model-dir .\models --device cuda
```

## Model weights (not in Release)

Download safetensors into a directory and pass `--model-dir`:

| File | Variant (`-m`) | Approx. size |
|------|----------------|--------------|
| `htdemucs.safetensors` | `htdemucs` | ~84 MB |
| `htdemucs_6s.safetensors` | `htdemucs_6s` | ~54 MB |
| `htdemucs_ft.safetensors` | `htdemucs_ft` | ~333 MB |

Common sources (pick one that still hosts the converted weights):

- Community safetensors mirrors of Facebook Research Demucs HTDemucs weights
- Convert from official Demucs checkpoints if you already have a conversion pipeline

Place them so paths look like:

```text
models/htdemucs.safetensors
models/htdemucs_6s.safetensors
models/htdemucs_ft.safetensors
```

## CLI

```text
demucs-native -i <input.wav> [options]

  -i, --input <path>       Input WAV (required)
  -o, --output <dir>       Output stems directory (default: ./stems/)
  -m, --model <name>       htdemucs | htdemucs_6s | htdemucs_ft
      --model-dir <dir>    Directory with *.safetensors (default: ./models)
      --device <name>       auto | cpu | cuda   (default: auto)
  -s, --stems <list>       e.g. vocals  or  vocals,drums
  -h, --help
```

`--device cuda` only works in binaries built with the `cuda` feature. CPU-only builds accept `auto` / `cpu`.

## Build from source

**Requirements:** Rust 1.75+ (edition 2021).

### CPU only (Windows / Linux / macOS)

```bash
cargo build --release --no-default-features
# binary: target/release/demucs-native[.exe]
```

### CUDA (Windows / Linux, CUDA Toolkit 12.x + `nvcc` on PATH)

```bash
cargo build --release --features cuda
```

`cudarc` uses `cuda-version-from-build-system` so the installed toolkit version is detected at build time. NVRTC loads `kernels.cu` and compiles for the GPU’s compute capability when you run.

## Features

| Cargo feature | Effect |
|---------------|--------|
| *(default)* | CPU engine + CLI (`--device auto\|cpu`) |
| `cuda` | Optional CUDA engine (`--device cuda`) |

## Project layout

```text
src/
  lib.rs            Public API
  backend.rs        Backend::Auto | Cpu | Cuda
  cpu_engine.rs     gemm + rayon path
  cuda_engine.rs    cudarc + cuBLAS + NVRTC
  cuda_ops.rs       CUDA ops / timing aggregate
  kernels/kernels.cu
  bin/demucs_native.rs
```

## CI / releases

Pushing a tag `v*` (or running **Actions → Release → Run workflow**) builds multi-platform archives and attaches them to a GitHub Release. CUDA jobs only **compile/link**; they do not execute on a GPU runner.

## License

MIT (see repository metadata). HTDemucs model weights remain under their original license (typically the Demucs / Meta Research terms).
