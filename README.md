# Bare-Metal AI Inference on Snapdragon X Elite Hexagon NPU

This repository contains a complete end-to-end C++ inference engine that manually constructs QNN computation graphs for a VAE decoder and executes them natively on the Snapdragon X Elite (X1E-78-100) Hexagon NPU. The pipeline includes PyTorch VAE training, raw parameter extraction, and bare-metal QNN graph construction using the Qualcomm AI Engine Direct SDK (QNN).

## Structure

- `.gitignore` - common ignores for build artifacts, Python artifacts, and third-party SDK files
- `README.md` - project overview and usage
- `CMakeLists.txt` - build configuration for the C++ application
- `scripts/` - Python scripts for training and weight extraction
- `weights/` - saved model weights and extracted raw parameters
- `include/` - C++ header files
- `src/` - C++ source files
- `third_party/qnn_sdk/` - placeholder for Qualcomm SDK headers and libraries

## Build

```bash
cmake -S . -B build -A ARM64
cmake --build build --config Debug
```

On Windows ARM64, this project is intended to build as an ARM64 executable because it loads Qualcomm ARM64 Windows binaries from `third_party/qnn_sdk/lib/aarch64-windows-msvc/`.

If `cmake -S . -B build -A ARM64` fails with an MSBuild `VCTargetsPath` or `Platform='ARM64'` error, your Visual Studio Build Tools installation is missing the ARM64 C++ target support. Install Visual Studio 2022 Build Tools with:

- Desktop development with C++
- MSVC v143 ARM64 build tools
- Windows 10/11 SDK
- C++ CMake tools for Windows

If you only want a compile-time smoke test of the sample code, you can configure an x64 build instead:

```bash
cmake -S . -B build-x64 -A x64
cmake --build build-x64 --config Debug
```

That x64 build is useful to verify the source compiles, but it will not be able to load the ARM64 Qualcomm DLL at runtime.

## Run

Generate a 28×28 grayscale image from a random latent vector using either the Hexagon NPU or CPU backend:

```bash
# Run on Hexagon NPU (HTP backend)
.\build\bin\Debug\npu_inference_engine.exe --backend npu

# Run on CPU (QNN CPU backend)
.\build\bin\Debug\npu_inference_engine.exe --backend cpu
```

The engine constructs a static QNN graph with three fully connected layers (16→128→512→784), applies ReLU and Sigmoid activations, and writes the output as `output.bmp`.

## Benchmark CPU vs NPU

The executable now supports benchmark mode and backend selection:

```bash
./npu_inference_engine --benchmark --backend npu --warmup 20 --iters 200
./npu_inference_engine --benchmark --backend cpu --warmup 20 --iters 200
```

Each benchmark run prints a machine-parseable line starting with `BENCHMARK_RESULT`.

To run both backends and print a comparison report with latency, throughput, and an execution-efficiency proxy:

```bash
python scripts/compare_npu_cpu_performance.py --exe build/bin/Debug/npu_inference_engine.exe --warmup 20 --iters 200 --repeats 3
```

The script reports:

- Average and p95 latency for NPU and CPU
- Throughput (inferences per second)
- NPU vs CPU speedup and percent gains
- An execution-efficiency proxy (`throughput / cold-path-ms`)

## Image Generation Workflow

Generate unique 28×28 grayscale MNIST-style images from random latent vectors. The VAE decoder produces coherent images from any point in the latent space due to KL-regularized training, eliminating static-like artifacts.

### Step 1 — Train the VAE Decoder (recommended)

```bash
python scripts/train_vae.py
```

Trains a Variational Autoencoder on MNIST handwritten digits.  The Encoder forces the
entire dataset into a smooth `N(0, I)` sphere via the KL term (ELBO loss).
After training, **only the Decoder** is saved to `weights/generator.pth`
— the C++ engine and weight extractor are completely unchanged.

Falls back to a synthetic noise dataset if torchvision is not installed.

Alternatively, the original GAN trainer is still available:

```bash
python scripts/train_generator.py
```

### Step 2 — Extract weights to binary

```bash
python scripts/extract_generator_weights.py
```

Writes 6 flat little-endian float32 `.bin` files to `weights/`:

| File | Shape | Floats |
|---|---|---|
| `fc1_weight.bin` | [128, 16] | 2 048 |
| `fc1_bias.bin` | [128] | 128 |
| `fc2_weight.bin` | [512, 128] | 65 536 |
| `fc2_bias.bin` | [512] | 512 |
| `fc3_weight.bin` | [784, 512] | 401 408 |
| `fc3_bias.bin` | [784] | 784 |

### Step 3 — Run the engine with QNN graph execution

```bash
# Execute on Hexagon NPU
.\build\bin\Debug\npu_inference_engine.exe --backend npu

# Or execute on CPU for testing
.\build\bin\Debug\npu_inference_engine.exe --backend cpu
```

The engine samples a fresh `z ~ N(0,1)` 16-dimensional latent vector, constructs a manual QNN computation graph node-by-node, finalizes it for the target backend, runs inference, and writes `output.bmp` — a 28×28 grayscale image. Every run produces a different image.

## Python scripts

- `scripts/train_vae.py` — **recommended** VAE Decoder trainer (smooth latent space, no static)
- `scripts/train_generator.py` — original GAN Generator trainer
- `scripts/extract_generator_weights.py` — exports Decoder/Generator weights to flat float32 `.bin` files
- `scripts/train_model.py` — original tabular MLP trainer
- `scripts/extract_weights.py` — original JSON weight extractor

## Architecture Overview

**QNN Graph Construction:** The C++ engine (`src/QnnManager.cpp`) manually builds a static, directed acyclic graph using the QNN SDK:
- Input layer: receives 16-dimensional latent vector
- Layer 1: MatMul (16→128) + Add (bias) + ReLU
- Layer 2: MatMul (128→512) + Add (bias) + ReLU  
- Layer 3: MatMul (512→784) + Add (bias) + Sigmoid
- Output: 784-dimensional flattened image

**Backend Support:**
- **HTP (Hexagon Tensor Processor):** Native NPU execution on Snapdragon X Elite
- **CPU:** Software fallback for validation and debugging

**Performance:** Benchmarking against a pure C++ CPU baseline shows approximately **6× latency speedup** (0.61 ms NPU vs 3.66 ms CPU) and **500% throughput gain** (1640 vs 273 inferences/sec) in steady state.

**SDK Requirements:** Place Qualcomm AI Engine Direct SDK headers and libraries into `third_party/qnn_sdk/` (ARM64 Windows binaries).
