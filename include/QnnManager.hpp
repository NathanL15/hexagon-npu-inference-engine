#pragma once

#include <string>

class QnnManager {
public:
    QnnManager();
    ~QnnManager();

    bool initialize();
    bool loadModel(const std::string &modelPath);
    bool runInference();

private:
    bool initialized_;
};
