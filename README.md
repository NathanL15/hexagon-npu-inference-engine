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
mkdir -p build && cd build
cmake ..
cmake --build .
```

## Run

```bash
./npu_inference_engine
```

## Python scripts

- `scripts/train_model.py` trains a small MLP using PyTorch and saves the model to `weights/model.pth`
- `scripts/extract_weights.py` extracts raw FP32 tensor values from the saved state dict and writes them to `weights/weights.json`

## Notes

Place Qualcomm AI Engine Direct SDK headers and libs into `third_party/qnn_sdk/` before integrating the actual NPU runtime.
