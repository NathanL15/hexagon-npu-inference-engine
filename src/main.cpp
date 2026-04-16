// Neural Canvas - Hexagon NPU Inference Engine
// =============================================
// Loads a trained GAN Generator's weights from flat float32 binary files,
// samples a random 16-dim latent vector, runs a pure-C++ forward pass:
//
//   z (16) -> Linear(16,128) -> ReLU
//          -> Linear(128,512) -> ReLU
//          -> Linear(512,784) -> Sigmoid
//          -> scale x255 -> 28x28 grayscale BMP
//
// Each run produces a unique image.  No Qualcomm SDK required for this step.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
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

// ---------------------------------------------------------------------------
// Dense layer: y[o] = sum_i(W[o*in + i] * x[i]) + b[o]
// W is stored row-major: W[out_features][in_features]
// ---------------------------------------------------------------------------

static std::vector<float> denseLayer(
    const std::vector<float> &x,
    const std::vector<float> &W,
    const std::vector<float> &b,
    int inFeatures,
    int outFeatures)
{
    std::vector<float> y(static_cast<size_t>(outFeatures));
    for (int o = 0; o < outFeatures; ++o) {
        float sum = b[static_cast<size_t>(o)];
        const float *row = W.data() + static_cast<size_t>(o) * inFeatures;
        for (int i = 0; i < inFeatures; ++i)
            sum += row[i] * x[static_cast<size_t>(i)];
        y[static_cast<size_t>(o)] = sum;
    }
    return y;
}

static void applyReLU(std::vector<float> &x)
{
    for (float &v : x)
        v = v > 0.0f ? v : 0.0f;
}

static void applySigmoid(std::vector<float> &x)
{
    for (float &v : x)
        v = 1.0f / (1.0f + std::exp(-v));
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

int main()
{
    std::cout << "=== Neural Canvas - Hexagon NPU Inference Engine ===" << std::endl;

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
    std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<float> stdNormal(0.0f, 1.0f);

    std::vector<float> z(16);
    for (float &v : z) v = stdNormal(rng);

    std::cout << "[OK] Latent z = [";
    for (int i = 0; i < 16; ++i)
        std::cout << (i ? ", " : "") << z[i];
    std::cout << "]\n";

    // ------------------------------------------------------------------
    // 3. Forward pass through the Generator
    // ------------------------------------------------------------------
    auto h1 = denseLayer(z,  fc1W, fc1b,  16, 128);
    applyReLU(h1);

    auto h2 = denseLayer(h1, fc2W, fc2b, 128, 512);
    applyReLU(h2);

    auto out = denseLayer(h2, fc3W, fc3b, 512, 784);
    applySigmoid(out);

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
    const std::string outPath = "output.bmp";
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
