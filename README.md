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

## Python scripts

- `scripts/train_model.py` trains a small MLP using PyTorch and saves the model to `weights/model.pth`
- `scripts/extract_weights.py` extracts raw FP32 tensor values from the saved state dict and writes them to `weights/weights.json`

## Notes

Place Qualcomm AI Engine Direct SDK headers and libs into `third_party/qnn_sdk/` before integrating the actual NPU runtime.
