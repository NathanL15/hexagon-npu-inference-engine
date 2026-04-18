#include "QnnManager.hpp"

#include <windows.h>

#include <array>
#include <iostream>
#include <memory>
#include <vector>

#include "QNN/QnnInterface.h"
#include "QNN/QnnOpDef.h"

namespace {

constexpr uint32_t INPUT_DIM = 16;
constexpr uint32_t FC1_DIM = 128;
constexpr uint32_t FC2_DIM = 512;
constexpr uint32_t FC3_DIM = 784;

void initTensorV1(Qnn_Tensor_t &tensor,
                  const char *name,
                  Qnn_TensorType_t type,
                  const std::vector<uint32_t> &dims,
                  void *data,
                  uint32_t dataBytes) {
    tensor = QNN_TENSOR_INIT;
    tensor.v1.name = name;
    tensor.v1.type = type;
    tensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
    tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tensor.v1.rank = (uint32_t)dims.size();
    tensor.v1.dimensions = (uint32_t*)dims.data();
    tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    if (type == QNN_TENSOR_TYPE_STATIC || type == QNN_TENSOR_TYPE_UPDATEABLE_STATIC) {
        tensor.v1.clientBuf.data = data;
        tensor.v1.clientBuf.dataSize = dataBytes;
    } else {
        tensor.v1.clientBuf.data = nullptr;
        tensor.v1.clientBuf.dataSize = 0;
    }
}

void initMatMulParams(Qnn_Param_t (&params)[2]) {
    params[0] = QNN_PARAM_INIT;
    params[0].paramType = QNN_PARAMTYPE_SCALAR;
    params[0].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    params[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    params[0].scalarParam.bool8Value = 0;

    params[1] = QNN_PARAM_INIT;
    params[1].paramType = QNN_PARAMTYPE_SCALAR;
    params[1].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    params[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    params[1].scalarParam.bool8Value = 1;
}

void initAddParams(Qnn_Param_t (&params)[1]) {
    params[0] = QNN_PARAM_INIT;
    params[0].paramType = QNN_PARAMTYPE_SCALAR;
    params[0].name = QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION;
    params[0].scalarParam.dataType = QNN_DATATYPE_UINT_32;
    params[0].scalarParam.uint32Value = QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD;
}

}  // namespace

struct QnnManager::Impl {
    HMODULE backendLib = nullptr;
    const QnnInterface_t *iface = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    std::string backendName = "unknown";

    std::vector<float> fc1W;
    std::vector<float> fc1B;
    std::vector<float> fc2W;
    std::vector<float> fc2B;
    std::vector<float> fc3W;
    std::vector<float> fc3B;

    std::vector<float> fc1MM;
    std::vector<float> fc1Add;
    std::vector<float> fc1Act;
    std::vector<float> fc2MM;
    std::vector<float> fc2Add;
    std::vector<float> fc2Act;
    std::vector<float> fc3MM;
    std::vector<float> outputBuf;

    std::vector<uint32_t> dim1x16   = {1, INPUT_DIM};
    std::vector<uint32_t> dim128x16 = {FC1_DIM, INPUT_DIM};
    std::vector<uint32_t> dim1x128  = {1, FC1_DIM};
    std::vector<uint32_t> dim512x128 = {FC2_DIM, FC1_DIM};
    std::vector<uint32_t> dim1x512  = {1, FC2_DIM};
    std::vector<uint32_t> dim784x512 = {FC3_DIM, FC2_DIM};
    std::vector<uint32_t> dim1x784  = {1, FC3_DIM};

    Qnn_Tensor_t inputTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc1WeightTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc1BiasTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc1MatMulTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc1AddTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc1OutTensor = QNN_TENSOR_INIT;

    Qnn_Tensor_t fc2WeightTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc2BiasTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc2MatMulTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc2AddTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc2OutTensor = QNN_TENSOR_INIT;

    Qnn_Tensor_t fc3WeightTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc3BiasTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t fc3MatMulTensor = QNN_TENSOR_INIT;
    Qnn_Tensor_t outputTensor = QNN_TENSOR_INIT;

    Qnn_Param_t matMulParams[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    Qnn_Param_t addParams[1] = {QNN_PARAM_INIT};

    bool graphBuilt = false;

    void releaseGraphOnly() {
        if (iface && iface->QNN_INTERFACE_VER_NAME.contextFree && context) {
            iface->QNN_INTERFACE_VER_NAME.contextFree(context, nullptr);
            context = nullptr;
            graph = nullptr;
            graphBuilt = false;
        }
        if (iface && iface->QNN_INTERFACE_VER_NAME.contextCreate && backend) {
            iface->QNN_INTERFACE_VER_NAME.contextCreate(backend, nullptr, nullptr, &context);
        }
    }

    void cleanup() {
        if (iface && iface->QNN_INTERFACE_VER_NAME.contextFree && context) {
            iface->QNN_INTERFACE_VER_NAME.contextFree(context, nullptr);
            context = nullptr;
            graph = nullptr;
        }
        if (iface && iface->QNN_INTERFACE_VER_NAME.backendFree && backend) {
            iface->QNN_INTERFACE_VER_NAME.backendFree(backend);
            backend = nullptr;
        }
        if (backendLib) {
            FreeLibrary(backendLib);
            backendLib = nullptr;
        }
        iface = nullptr;
        graphBuilt = false;
    }
};

QnnManager::QnnManager() : impl_(std::make_unique<Impl>()), initialized_(false) {}

QnnManager::~QnnManager() {
    impl_->cleanup();
}

bool QnnManager::initialize(Backend backend) {
    const char *dllPath = nullptr;
    if (backend == Backend::Cpu) {
        dllPath = "third_party/qnn_sdk/lib/aarch64-windows-msvc/QnnCpu.dll";
        impl_->backendName = "cpu";
    } else {
        dllPath = "third_party/qnn_sdk/lib/aarch64-windows-msvc/QnnHtp.dll";
        impl_->backendName = "npu";
    }

    std::cout << "Initializing QNN manager (" << impl_->backendName << " backend)...\n";

    impl_->backendLib = LoadLibraryA(dllPath);
    if (!impl_->backendLib) {
        std::cerr << "QNN init failed: could not load backend DLL at " << dllPath << "\n";
        return false;
    }

    using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    auto getProviders = reinterpret_cast<GetProvidersFn>(
        GetProcAddress(impl_->backendLib, "QnnInterface_getProviders"));
    if (!getProviders) {
        std::cerr << "QNN init failed: QnnInterface_getProviders symbol missing\n";
        impl_->cleanup();
        return false;
    }

    const QnnInterface_t **providers = nullptr;
    uint32_t numProviders = 0;
    auto rc = getProviders(&providers, &numProviders);
    if (rc != QNN_SUCCESS || !providers || numProviders == 0) {
        std::cerr << "QNN init failed: no interface providers available\n";
        impl_->cleanup();
        return false;
    }

    impl_->iface = providers[0];
    if (!impl_->iface || !impl_->iface->QNN_INTERFACE_VER_NAME.backendCreate) {
        std::cerr << "QNN init failed: backendCreate entrypoint unavailable\n";
        impl_->cleanup();
        return false;
    }

    rc = impl_->iface->QNN_INTERFACE_VER_NAME.backendCreate(nullptr, nullptr, &impl_->backend);
    if (rc != QNN_SUCCESS || !impl_->backend) {
        std::cerr << "QNN init failed: backendCreate returned " << rc << "\n";
        impl_->cleanup();
        return false;
    }

    rc = impl_->iface->QNN_INTERFACE_VER_NAME.contextCreate(
        impl_->backend, nullptr, nullptr, &impl_->context);
    if (rc != QNN_SUCCESS || !impl_->context) {
        std::cerr << "QNN init failed: contextCreate returned " << rc << "\n";
        impl_->cleanup();
        return false;
    }

    initialized_ = true;
    std::cout << "QNN backend/context initialized (" << impl_->backendName << ").\n";
    return true;
}

bool QnnManager::buildGraph(const std::vector<float> &fc1Weight,
                            const std::vector<float> &fc1Bias,
                            const std::vector<float> &fc2Weight,
                            const std::vector<float> &fc2Bias,
                            const std::vector<float> &fc3Weight,
                            const std::vector<float> &fc3Bias) {
    if (!initialized_) {
        std::cerr << "QnnManager not initialized.\n";
        return false;
    }

    if (fc1Weight.size() != FC1_DIM * INPUT_DIM || fc1Bias.size() != FC1_DIM ||
        fc2Weight.size() != FC2_DIM * FC1_DIM || fc2Bias.size() != FC2_DIM ||
        fc3Weight.size() != FC3_DIM * FC2_DIM || fc3Bias.size() != FC3_DIM) {
        std::cerr << "QNN build failed: invalid tensor sizes provided.\n";
        return false;
    }

    impl_->fc1W = fc1Weight;
    impl_->fc1B = fc1Bias;
    impl_->fc2W = fc2Weight;
    impl_->fc2B = fc2Bias;
    impl_->fc3W = fc3Weight;
    impl_->fc3B = fc3Bias;

    impl_->fc1MM.assign(FC1_DIM, 0.0f);
    impl_->fc1Add.assign(FC1_DIM, 0.0f);
    impl_->fc1Act.assign(FC1_DIM, 0.0f);
    impl_->fc2MM.assign(FC2_DIM, 0.0f);
    impl_->fc2Add.assign(FC2_DIM, 0.0f);
    impl_->fc2Act.assign(FC2_DIM, 0.0f);
    impl_->fc3MM.assign(FC3_DIM, 0.0f);
    impl_->outputBuf.assign(FC3_DIM, 0.0f);

    if (impl_->graphBuilt) {
        impl_->releaseGraphOnly();
        if (!impl_->context) {
            std::cerr << "QNN build failed: could not recreate context.\n";
            return false;
        }
    }

    auto &ifc = impl_->iface->QNN_INTERFACE_VER_NAME;
    auto rc = ifc.graphCreate(impl_->context, "vae_decoder_full", nullptr, &impl_->graph);
    if (rc != QNN_SUCCESS || !impl_->graph) {
        std::cerr << "QNN build failed: graphCreate returned " << rc << "\n";
        return false;
    }

    auto bytes = [](auto &v) { return (uint32_t)(v.size() * 4); };

    initTensorV1(impl_->inputTensor, "latent_input", QNN_TENSOR_TYPE_APP_WRITE, impl_->dim1x16, nullptr, 0);

    initTensorV1(impl_->fc1WeightTensor, "fc1_weight", QNN_TENSOR_TYPE_STATIC, impl_->dim128x16, impl_->fc1W.data(), bytes(impl_->fc1W));
    initTensorV1(impl_->fc1BiasTensor,   "fc1_bias",   QNN_TENSOR_TYPE_STATIC, impl_->dim1x128,  impl_->fc1B.data(), bytes(impl_->fc1B));
    initTensorV1(impl_->fc1MatMulTensor, "fc1_matmul_out", QNN_TENSOR_TYPE_NATIVE, impl_->dim1x128, impl_->fc1MM.data(),  bytes(impl_->fc1MM));
    initTensorV1(impl_->fc1AddTensor,    "fc1_add_out",    QNN_TENSOR_TYPE_NATIVE, impl_->dim1x128, impl_->fc1Add.data(), bytes(impl_->fc1Add));
    initTensorV1(impl_->fc1OutTensor,    "fc1_relu_out",   QNN_TENSOR_TYPE_NATIVE, impl_->dim1x128, impl_->fc1Act.data(), bytes(impl_->fc1Act));

    initTensorV1(impl_->fc2WeightTensor, "fc2_weight", QNN_TENSOR_TYPE_STATIC, impl_->dim512x128, impl_->fc2W.data(), bytes(impl_->fc2W));
    initTensorV1(impl_->fc2BiasTensor,   "fc2_bias",   QNN_TENSOR_TYPE_STATIC, impl_->dim1x512,   impl_->fc2B.data(), bytes(impl_->fc2B));
    initTensorV1(impl_->fc2MatMulTensor, "fc2_matmul_out", QNN_TENSOR_TYPE_NATIVE, impl_->dim1x512, impl_->fc2MM.data(),  bytes(impl_->fc2MM));
    initTensorV1(impl_->fc2AddTensor,    "fc2_add_out",    QNN_TENSOR_TYPE_NATIVE, impl_->dim1x512, impl_->fc2Add.data(), bytes(impl_->fc2Add));
    initTensorV1(impl_->fc2OutTensor,    "fc2_relu_out",   QNN_TENSOR_TYPE_NATIVE, impl_->dim1x512, impl_->fc2Act.data(), bytes(impl_->fc2Act));

    initTensorV1(impl_->fc3WeightTensor, "fc3_weight", QNN_TENSOR_TYPE_STATIC, impl_->dim784x512, impl_->fc3W.data(), bytes(impl_->fc3W));
    initTensorV1(impl_->fc3BiasTensor,   "fc3_bias",   QNN_TENSOR_TYPE_STATIC, impl_->dim1x784,   impl_->fc3B.data(), bytes(impl_->fc3B));
    initTensorV1(impl_->fc3MatMulTensor, "fc3_matmul_out", QNN_TENSOR_TYPE_NATIVE, impl_->dim1x784, impl_->fc3MM.data(),    bytes(impl_->fc3MM));
    initTensorV1(impl_->outputTensor,    "decoder_output", QNN_TENSOR_TYPE_APP_READ, impl_->dim1x784, impl_->outputBuf.data(), bytes(impl_->outputBuf));

    Qnn_Tensor_t *allTensors[] = {
        &impl_->inputTensor,
        &impl_->fc1WeightTensor,
        &impl_->fc1BiasTensor,
        &impl_->fc1MatMulTensor,
        &impl_->fc1AddTensor,
        &impl_->fc1OutTensor,
        &impl_->fc2WeightTensor,
        &impl_->fc2BiasTensor,
        &impl_->fc2MatMulTensor,
        &impl_->fc2AddTensor,
        &impl_->fc2OutTensor,
        &impl_->fc3WeightTensor,
        &impl_->fc3BiasTensor,
        &impl_->fc3MatMulTensor,
        &impl_->outputTensor};

    for (Qnn_Tensor_t *tensor : allTensors) {
        rc = ifc.tensorCreateGraphTensor(impl_->graph, tensor);
        if (rc != QNN_SUCCESS) {
            std::cerr << "QNN build failed: tensorCreateGraphTensor returned " << rc
                      << " for tensor " << (tensor->v1.name ? tensor->v1.name : "<unnamed>")
                      << "\n";
            return false;
        }
    }

    initMatMulParams(impl_->matMulParams);
    initAddParams(impl_->addParams);

    auto addNode = [&](const char *nodeName, const char *typeName,
                       Qnn_Param_t *params, uint32_t numParams,
                       Qnn_Tensor_t *inputs, uint32_t numInputs,
                       Qnn_Tensor_t *outputs,
                       uint32_t numOutputs) -> bool {
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.v1.name = nodeName;
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName = typeName;
        op.v1.numOfParams = numParams;
        op.v1.params = params;
        op.v1.numOfInputs = numInputs;
        op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = numOutputs;
        op.v1.outputTensors = outputs;
        auto nodeRc = ifc.graphAddNode(impl_->graph, op);
        if (nodeRc != QNN_SUCCESS) {
            std::cerr << "QNN build failed: graphAddNode(" << nodeName << ") returned "
                      << nodeRc << "\n";
            return false;
        }
        return true;
    };

    Qnn_Tensor_t fc1MatMulIn[2] = {impl_->inputTensor, impl_->fc1WeightTensor};
    Qnn_Tensor_t fc1MatMulOut[1] = {impl_->fc1MatMulTensor};
    if (!addNode("fc1_matmul", QNN_OP_MAT_MUL, impl_->matMulParams, 2, fc1MatMulIn, 2, fc1MatMulOut, 1)) {
        return false;
    }

    Qnn_Tensor_t fc1AddIn[2] = {impl_->fc1MatMulTensor, impl_->fc1BiasTensor};
    Qnn_Tensor_t fc1AddOut[1] = {impl_->fc1AddTensor};
    if (!addNode("fc1_add", QNN_OP_ELEMENT_WISE_BINARY, impl_->addParams, 1, fc1AddIn, 2, fc1AddOut, 1)) {
        return false;
    }

    Qnn_Tensor_t fc1ReluIn[1] = {impl_->fc1AddTensor};
    Qnn_Tensor_t fc1ReluOut[1] = {impl_->fc1OutTensor};
    if (!addNode("fc1_relu", QNN_OP_RELU, nullptr, 0, fc1ReluIn, 1, fc1ReluOut, 1)) {
        return false;
    }

    Qnn_Tensor_t fc2MatMulIn[2] = {impl_->fc1OutTensor, impl_->fc2WeightTensor};
    Qnn_Tensor_t fc2MatMulOut[1] = {impl_->fc2MatMulTensor};
    if (!addNode("fc2_matmul", QNN_OP_MAT_MUL, impl_->matMulParams, 2, fc2MatMulIn, 2, fc2MatMulOut, 1)) {
        return false;
    }

    Qnn_Tensor_t fc2AddIn[2] = {impl_->fc2MatMulTensor, impl_->fc2BiasTensor};
    Qnn_Tensor_t fc2AddOut[1] = {impl_->fc2AddTensor};
    if (!addNode("fc2_add", QNN_OP_ELEMENT_WISE_BINARY, impl_->addParams, 1, fc2AddIn, 2, fc2AddOut, 1)) {
        return false;
    }

    Qnn_Tensor_t fc2ReluIn[1] = {impl_->fc2AddTensor};
    Qnn_Tensor_t fc2ReluOut[1] = {impl_->fc2OutTensor};
    if (!addNode("fc2_relu", QNN_OP_RELU, nullptr, 0, fc2ReluIn, 1, fc2ReluOut, 1)) {
        return false;
    }

    Qnn_Tensor_t fc3MatMulIn[2] = {impl_->fc2OutTensor, impl_->fc3WeightTensor};
    Qnn_Tensor_t fc3MatMulOut[1] = {impl_->fc3MatMulTensor};
    if (!addNode("fc3_matmul", QNN_OP_MAT_MUL, impl_->matMulParams, 2, fc3MatMulIn, 2, fc3MatMulOut, 1)) {
        return false;
    }

    Qnn_Tensor_t fc3AddIn[2] = {impl_->fc3MatMulTensor, impl_->fc3BiasTensor};
    Qnn_Tensor_t fc3AddOut[1] = {impl_->outputTensor};
    if (!addNode("fc3_add", QNN_OP_ELEMENT_WISE_BINARY, impl_->addParams, 1, fc3AddIn, 2, fc3AddOut, 1)) {
        return false;
    }

    Qnn_Tensor_t fc3SigmoidIn[1] = {impl_->outputTensor};
    Qnn_Tensor_t fc3SigmoidOut[1] = {impl_->outputTensor};
    if (!addNode("fc3_sigmoid", QNN_OP_SIGMOID, nullptr, 0, fc3SigmoidIn, 1, fc3SigmoidOut, 1)) {
        return false;
    }

    rc = ifc.graphFinalize(impl_->graph, nullptr, nullptr);
    if (rc != QNN_SUCCESS) {
        std::cerr << "QNN build failed: graphFinalize returned " << rc << "\n";
        return false;
    }

    impl_->graphBuilt = true;
    std::cout << "QNN full decoder graph finalized on " << impl_->backendName << " backend.\n";
    return true;
}

bool QnnManager::runInference(const std::vector<float> &input, std::vector<float> &output) {
    if (!initialized_) {
        std::cerr << "QnnManager not initialized.\n";
        return false;
    }
    if (!impl_->graphBuilt || !impl_->graph) {
        std::cerr << "QNN graph is not built. Call buildGraph() first.\n";
        return false;
    }
    if (input.size() != INPUT_DIM) {
        std::cerr << "QNN run failed: expected input size 16, got " << input.size() << "\n";
        return false;
    }

    auto runInput = impl_->inputTensor;
    runInput.v1.clientBuf.data = (float*)input.data();
    runInput.v1.clientBuf.dataSize = (uint32_t)(input.size() * sizeof(float));

    auto runOutput = impl_->outputTensor;
    runOutput.v1.clientBuf.data = impl_->outputBuf.data();
    runOutput.v1.clientBuf.dataSize = (uint32_t)(impl_->outputBuf.size() * sizeof(float));

    auto rc = impl_->iface->QNN_INTERFACE_VER_NAME.graphExecute(
        impl_->graph, &runInput, 1, &runOutput, 1, nullptr, nullptr);
    if (rc != QNN_SUCCESS) {
        std::cerr << "QNN run failed: graphExecute returned " << rc << "\n";
        return false;
    }

    output = impl_->outputBuf;
    return true;
}

std::string QnnManager::backendName() const {
    return impl_->backendName;
}
