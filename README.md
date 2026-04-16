# NPU Inference Engine

This repository contains a minimal C++ application and supporting Python scripts for training a tabular MLP, extracting model weights, and preparing a Qualcomm NPU inference engine integration.

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

```bash
./npu_inference_engine
```

## Neural Canvas workflow

Generate a unique 28x28 grayscale image from a random latent vector.  The
engine uses a VAE-trained Decoder whose latent space is a smooth Gaussian ball,
so every random point produces a coherent image — no static.

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

### Step 3 — Run the engine

```bash
.\build\bin\Debug\npu_inference_engine.exe
```

The engine samples a fresh `z ~ N(0,1)` 16-dim latent vector, runs the forward
pass in pure C++, and writes `output.bmp` — a 28x28 grayscale image.  Every
run produces a different image.

## Python scripts

- `scripts/train_vae.py` — **recommended** VAE Decoder trainer (smooth latent space, no static)
- `scripts/train_generator.py` — original GAN Generator trainer
- `scripts/extract_generator_weights.py` — exports Decoder/Generator weights to flat float32 `.bin` files
- `scripts/train_model.py` — original tabular MLP trainer
- `scripts/extract_weights.py` — original JSON weight extractor

## Notes

Place Qualcomm AI Engine Direct SDK headers and libs into `third_party/qnn_sdk/`
before integrating the actual NPU runtime.  The current C++ engine performs the
forward pass in software; the weight binary format is already compatible with
QNN `Gemm` operator inputs.
