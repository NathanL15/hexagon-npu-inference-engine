// Neural Canvas - Hexagon NPU Inference Engine
// =============================================
// Loads trained decoder weights from flat float32 binary files,
// samples a random 16-dim latent vector, and runs the full decoder graph
// on Qualcomm QNN/HTP.
//
//   z (16) -> Linear(16,128) -> ReLU
//          -> Linear(128,512) -> ReLU
//          -> Linear(512,784) -> Sigmoid
//          -> scale x255 -> 28x28 grayscale BMP
//
// Each run produces a unique image. There is no CPU neural-network fallback.

#include "QnnManager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

static std::vector<float> loadBin(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open weight file: " + path);

    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    if (bytes == 0 || bytes % sizeof(float) != 0)
        throw std::runtime_error("Weight file has unexpected size: " + path);

    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char *>(data.data()), bytes);
    if (!f)
        throw std::runtime_error("Read error on weight file: " + path);

    return data;
}

struct CliOptions {
    std::string backend = "npu";
    bool benchmark = false;
    int warmup = 20;
    int iters = 200;
    std::string outputPath = "output.bmp";
};

static void printUsage(const char *exeName)
{
    std::cout
        << "Usage: " << exeName << " [options]\n"
        << "\nOptions:\n"
        << "  --backend <npu|cpu>  Select QNN backend (default: npu)\n"
        << "  --benchmark          Run latency benchmark instead of image generation\n"
        << "  --warmup <N>         Warmup iterations for benchmark (default: 20)\n"
        << "  --iters <N>          Timed iterations for benchmark (default: 200)\n"
        << "  --output <path>      Output BMP path for non-benchmark mode\n"
        << "  --help               Show this help\n";
}

static bool parseArgs(int argc, char **argv, CliOptions &opts)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        }
        if (arg == "--benchmark") {
            opts.benchmark = true;
            continue;
        }
        if (arg == "--backend") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --backend\n";
                return false;
            }
            opts.backend = argv[++i];
            if (opts.backend != "npu" && opts.backend != "cpu") {
                std::cerr << "Invalid backend: " << opts.backend << " (expected npu or cpu)\n";
                return false;
            }
            continue;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --warmup\n";
                return false;
            }
            opts.warmup = std::atoi(argv[++i]);
            if (opts.warmup < 0) {
                std::cerr << "--warmup must be >= 0\n";
                return false;
            }
            continue;
        }
        if (arg == "--iters") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --iters\n";
                return false;
            }
            opts.iters = std::atoi(argv[++i]);
            if (opts.iters <= 0) {
                std::cerr << "--iters must be > 0\n";
                return false;
            }
            continue;
        }
        if (arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --output\n";
                return false;
            }
            opts.outputPath = argv[++i];
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        return false;
    }
    return true;
}

static QnnManager::Backend parseBackend(const std::string &backend)
{
    return backend == "cpu" ? QnnManager::Backend::Cpu : QnnManager::Backend::Htp;
}

static double elapsedMs(const std::chrono::high_resolution_clock::time_point &start,
                        const std::chrono::high_resolution_clock::time_point &end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void denseLayer(const std::vector<float> &input,
                       const std::vector<float> &weight,
                       const std::vector<float> &bias,
                       int inDim,
                       int outDim,
                       std::vector<float> &output)
{
    output.assign(static_cast<size_t>(outDim), 0.0f);
    for (int o = 0; o < outDim; ++o) {
        float sum = bias[static_cast<size_t>(o)];
        const size_t rowOffset = static_cast<size_t>(o * inDim);
        for (int i = 0; i < inDim; ++i) {
            sum += weight[rowOffset + static_cast<size_t>(i)] * input[static_cast<size_t>(i)];
        }
        output[static_cast<size_t>(o)] = sum;
    }
}

static void reluInplace(std::vector<float> &x)
{
    for (float &v : x) {
        v = std::max(0.0f, v);
    }
}

static void sigmoidInplace(std::vector<float> &x)
{
    for (float &v : x) {
        v = 1.0f / (1.0f + std::exp(-v));
    }
}

static void runCpuDecoder(const std::vector<float> &z,
                          const std::vector<float> &fc1W,
                          const std::vector<float> &fc1b,
                          const std::vector<float> &fc2W,
                          const std::vector<float> &fc2b,
                          const std::vector<float> &fc3W,
                          const std::vector<float> &fc3b,
                          std::vector<float> &out)
{
    std::vector<float> fc1;
    std::vector<float> fc2;

    denseLayer(z, fc1W, fc1b, 16, 128, fc1);
    reluInplace(fc1);
    denseLayer(fc1, fc2W, fc2b, 128, 512, fc2);
    reluInplace(fc2);
    denseLayer(fc2, fc3W, fc3b, 512, 784, out);
    sigmoidInplace(out);
}

// ---------------------------------------------------------------------------
// BMP writer  (24-bit RGB, grayscale via R=G=B; top-down via negative height)
// ---------------------------------------------------------------------------

static void writeBMP(
    const std::string &path,
    const std::vector<uint8_t> &pixels,
    int width,
    int height)
{
    // Each row must be padded to a 4-byte boundary.
    // 28 * 3 = 84 bytes -> 84 % 4 = 0, so no padding needed for 28-pixel rows.
    const int rowStride    = ((width * 3 + 3) / 4) * 4;
    const int pixelBytes   = rowStride * height;
    const int fileSize     = 14 + 40 + pixelBytes;
    const int pixelOffset  = 14 + 40;

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot create output file: " + path);

    // ---- BITMAPFILEHEADER (14 bytes) ----
    auto put32 = [&](int32_t v) {
        f.put(static_cast<char>(v & 0xFF));
        f.put(static_cast<char>((v >> 8) & 0xFF));
        f.put(static_cast<char>((v >> 16) & 0xFF));
        f.put(static_cast<char>((v >> 24) & 0xFF));
    };
    auto put16 = [&](int16_t v) {
        f.put(static_cast<char>(v & 0xFF));
        f.put(static_cast<char>((v >> 8) & 0xFF));
    };

    f.put('B'); f.put('M');
    put32(fileSize);
    put32(0);           // reserved
    put32(pixelOffset);

    // ---- BITMAPINFOHEADER (40 bytes) ----
    put32(40);           // header size
    put32(width);
    put32(-height);      // negative = top-down row order
    put16(1);            // color planes
    put16(24);           // bits per pixel
    put32(0);            // BI_RGB (no compression)
    put32(pixelBytes);
    put32(2835);         // ~72 DPI horizontal
    put32(2835);         // ~72 DPI vertical
    put32(0);            // colors in palette
    put32(0);            // important colors

    // ---- Pixel data (BGR order) ----
    std::vector<uint8_t> row(static_cast<size_t>(rowStride), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uint8_t v = pixels[static_cast<size_t>(y * width + x)];
            row[static_cast<size_t>(x * 3 + 0)] = v; // B
            row[static_cast<size_t>(x * 3 + 1)] = v; // G
            row[static_cast<size_t>(x * 3 + 2)] = v; // R
        }
        f.write(reinterpret_cast<const char *>(row.data()), rowStride);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    CliOptions options;
    if (!parseArgs(argc, argv, options)) {
        return 1;
    }

    std::cout << "=== Neural Canvas - Hexagon NPU Inference Engine ===" << std::endl;
    std::cout << "Selected backend: " << options.backend << "\n";

    // ------------------------------------------------------------------
    // 1. Load the six weight tensors
    // ------------------------------------------------------------------
    const std::string wDir = "weights/";

    std::vector<float> fc1W, fc1b, fc2W, fc2b, fc3W, fc3b;
    try {
        fc1W = loadBin(wDir + "fc1_weight.bin"); // [128, 16]
        fc1b = loadBin(wDir + "fc1_bias.bin");   // [128]
        fc2W = loadBin(wDir + "fc2_weight.bin"); // [512, 128]
        fc2b = loadBin(wDir + "fc2_bias.bin");   // [512]
        fc3W = loadBin(wDir + "fc3_weight.bin"); // [784, 512]
        fc3b = loadBin(wDir + "fc3_bias.bin");   // [784]
    } catch (const std::exception &e) {
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        std::cerr << "Run the following scripts first:\n"
                  << "  python scripts/train_generator.py\n"
                  << "  python scripts/extract_generator_weights.py\n";
        return 1;
    }
    std::cout << "[OK] Weights loaded." << std::endl;

    // ------------------------------------------------------------------
    // 2. Sample a random 16-dim latent vector z ~ N(0, 1)
    // ------------------------------------------------------------------
    std::mt19937 rng{options.benchmark ? 12345u : std::random_device{}()};
    std::normal_distribution<float> stdNormal(0.0f, 1.0f);

    std::vector<float> z(16);
    for (float &v : z) v = stdNormal(rng);

    if (!options.benchmark) {
        std::cout << "[OK] Latent z = [";
        for (int i = 0; i < 16; ++i)
            std::cout << (i ? ", " : "") << z[i];
        std::cout << "]\n";
    }

    // ------------------------------------------------------------------
    // 3. Forward pass through selected backend (QNN NPU or native CPU)
    // ------------------------------------------------------------------
    std::vector<float> out;
    double initMs = 0.0;
    double buildMs = 0.0;
    double coldMs = 0.0;

    if (options.backend == "cpu") {
        const auto coldStart = std::chrono::high_resolution_clock::now();
        runCpuDecoder(z, fc1W, fc1b, fc2W, fc2b, fc3W, fc3b, out);
        const auto coldEnd = std::chrono::high_resolution_clock::now();
        coldMs = elapsedMs(coldStart, coldEnd);

        if (options.benchmark) {
            for (int i = 0; i < options.warmup; ++i) {
                runCpuDecoder(z, fc1W, fc1b, fc2W, fc2b, fc3W, fc3b, out);
            }

            std::vector<double> latencies;
            latencies.reserve(static_cast<size_t>(options.iters));
            for (int i = 0; i < options.iters; ++i) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                runCpuDecoder(z, fc1W, fc1b, fc2W, fc2b, fc3W, fc3b, out);
                const auto t1 = std::chrono::high_resolution_clock::now();
                latencies.push_back(elapsedMs(t0, t1));
            }

            std::sort(latencies.begin(), latencies.end());
            const double minMs = latencies.front();
            const double maxMs = latencies.back();
            const double sumMs = std::accumulate(latencies.begin(), latencies.end(), 0.0);
            const double avgMs = sumMs / static_cast<double>(latencies.size());
            const double p50Ms = latencies[latencies.size() / 2];
            const size_t p95Idx = static_cast<size_t>(std::ceil(0.95 * latencies.size())) - 1;
            const double p95Ms = latencies[std::min(p95Idx, latencies.size() - 1)];
            const double throughput = 1000.0 / avgMs;
            const double checksum = std::accumulate(out.begin(), out.end(), 0.0);

            std::cout << std::fixed << std::setprecision(4);
            std::cout << "BENCHMARK_RESULT"
                      << " backend=cpu"
                      << " init_ms=" << initMs
                      << " build_ms=" << buildMs
                      << " cold_ms=" << coldMs
                      << " warmup=" << options.warmup
                      << " iters=" << options.iters
                      << " avg_ms=" << avgMs
                      << " p50_ms=" << p50Ms
                      << " p95_ms=" << p95Ms
                      << " min_ms=" << minMs
                      << " max_ms=" << maxMs
                      << " throughput_fps=" << throughput
                      << " checksum=" << checksum
                      << "\n";
            return 0;
        }

        std::cout << "[OK] Full decoder executed on native CPU path.\n";
    } else {
        QnnManager qnn;
        const auto initStart = std::chrono::high_resolution_clock::now();
        if (!qnn.initialize(parseBackend(options.backend))) {
            std::cerr << "[ERROR] Failed to initialize QNN backend.\n";
            return 1;
        }
        const auto initEnd = std::chrono::high_resolution_clock::now();

        const auto buildStart = std::chrono::high_resolution_clock::now();
        if (!qnn.buildGraph(fc1W, fc1b, fc2W, fc2b, fc3W, fc3b)) {
            std::cerr << "[ERROR] Failed to build full QNN decoder graph.\n";
            return 1;
        }
        const auto buildEnd = std::chrono::high_resolution_clock::now();

        const auto coldStart = std::chrono::high_resolution_clock::now();
        if (!qnn.runInference(z, out) || out.size() != 784) {
            std::cerr << "[ERROR] Failed to execute full QNN decoder inference.\n";
            return 1;
        }
        const auto coldEnd = std::chrono::high_resolution_clock::now();

        initMs = elapsedMs(initStart, initEnd);
        buildMs = elapsedMs(buildStart, buildEnd);
        coldMs = elapsedMs(coldStart, coldEnd);

        if (options.benchmark) {
            for (int i = 0; i < options.warmup; ++i) {
                if (!qnn.runInference(z, out)) {
                    std::cerr << "[ERROR] Warmup iteration failed at index " << i << "\n";
                    return 1;
                }
            }

            std::vector<double> latencies;
            latencies.reserve(static_cast<size_t>(options.iters));

            for (int i = 0; i < options.iters; ++i) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                if (!qnn.runInference(z, out)) {
                    std::cerr << "[ERROR] Timed iteration failed at index " << i << "\n";
                    return 1;
                }
                const auto t1 = std::chrono::high_resolution_clock::now();
                latencies.push_back(elapsedMs(t0, t1));
            }

            std::sort(latencies.begin(), latencies.end());
            const double minMs = latencies.front();
            const double maxMs = latencies.back();
            const double sumMs = std::accumulate(latencies.begin(), latencies.end(), 0.0);
            const double avgMs = sumMs / static_cast<double>(latencies.size());
            const double p50Ms = latencies[latencies.size() / 2];
            const size_t p95Idx = static_cast<size_t>(std::ceil(0.95 * latencies.size())) - 1;
            const double p95Ms = latencies[std::min(p95Idx, latencies.size() - 1)];
            const double throughput = 1000.0 / avgMs;
            const double checksum = std::accumulate(out.begin(), out.end(), 0.0);

            std::cout << std::fixed << std::setprecision(4);
            std::cout << "BENCHMARK_RESULT"
                      << " backend=" << qnn.backendName()
                      << " init_ms=" << initMs
                      << " build_ms=" << buildMs
                      << " cold_ms=" << coldMs
                      << " warmup=" << options.warmup
                      << " iters=" << options.iters
                      << " avg_ms=" << avgMs
                      << " p50_ms=" << p50Ms
                      << " p95_ms=" << p95Ms
                      << " min_ms=" << minMs
                      << " max_ms=" << maxMs
                      << " throughput_fps=" << throughput
                      << " checksum=" << checksum
                      << "\n";
            return 0;
        }

        std::cout << "[OK] Full decoder executed on Qualcomm QNN backend.\n";
    }

    const float minVal = *std::min_element(out.begin(), out.end());
    const float maxVal = *std::max_element(out.begin(), out.end());
    std::cout << "[OK] Forward pass complete.  Output range: ["
              << minVal << ", " << maxVal << "]\n";

    // ------------------------------------------------------------------
    // 4. Convert float [0,1] -> uint8 [0,255]
    // ------------------------------------------------------------------
    std::vector<uint8_t> pixels(784);
    for (int i = 0; i < 784; ++i)
        pixels[static_cast<size_t>(i)] =
            static_cast<uint8_t>(out[static_cast<size_t>(i)] * 255.0f);

    // ------------------------------------------------------------------
    // 5. Write 28x28 grayscale BMP
    // ------------------------------------------------------------------
    const std::string outPath = options.outputPath;
    try {
        writeBMP(outPath, pixels, 28, 28);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] Image saved: " << outPath << "  (28x28 grayscale BMP)\n";
    std::cout << "Run again for a completely different image.\n";
    return 0;
}
