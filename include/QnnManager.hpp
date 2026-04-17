#pragma once

#include <memory>
#include <string>
#include <vector>

class QnnManager {
public:
    QnnManager();
    ~QnnManager();

    bool initialize();
    bool buildGraph(const std::vector<float> &fc1Weight,
                    const std::vector<float> &fc1Bias,
                    const std::vector<float> &fc2Weight,
                    const std::vector<float> &fc2Bias,
                    const std::vector<float> &fc3Weight,
                    const std::vector<float> &fc3Bias);
    bool runInference(const std::vector<float> &input, std::vector<float> &output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool initialized_;
};
