#include <iostream>
#include "QnnManager.hpp"

int main() {
    std::cout << "Starting NPU inference engine...\n";

    QnnManager manager;
    if (!manager.initialize()) {
        std::cerr << "Failed to initialize QnnManager.\n";
        return 1;
    }

    if (!manager.loadModel("weights/model.pth")) {
        std::cerr << "Failed to load model.\n";
        return 1;
    }

    if (!manager.runInference()) {
        std::cerr << "Inference failed.\n";
        return 1;
    }

    std::cout << "Inference completed successfully.\n";
    return 0;
}
