#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "coreml_backend.h"
#include "cpu_conv.h"
#include "cpu_fast.h"
#include "mps_backend.h"
#include "weights.h"

static const std::string MODEL_DIR = "models/";
static const int SIZES[4][2] = {{256, 256}, {512, 512}, {1024, 1024}, {1080, 1920}};

static std::string model_path(int h, int w, int bits) {
    return MODEL_DIR + "sr_" + std::to_string(h) + "x" + std::to_string(w) + "_fp" + std::to_string(bits) + ".mlpackage";
}

static double maxDiff(const Tensor& a, const Tensor& b) {
    double m = 0;
    for (size_t i = 0; i < a.data.size(); i++) {
        double d = std::fabs(a.data[i] - b.data[i]);
        if (d > m) m = d;
    }
    return m;
}

static double psnr(const Tensor &a, const Tensor &b)
{
    double mse = 0;
    for (size_t i = 0; i < a.data.size(); i++) {
        double d = a.data[i] - b.data[i];
        mse += d * d;
    }
    mse /= a.data.size();
    if (mse == 0) return 999; // identical
    return 10 * std::log10(1.0 / mse);
}

static Tensor randInput(int h, int w) {
    Tensor t(1, h, w);
    srand(1);
    for (float& v : t.data) v = (float)rand() / (float)RAND_MAX;
    return t;
}

static void check(const char* name, const Tensor& out, const Tensor& ref) {
    if (out.data.size() != ref.data.size()) {
        printf("%-34s WRONG SHAPE\n", name);
        return;
    }
    printf("%-34s max abs err %.2e   psnr %6.1f dB\n", name, maxDiff(out, ref), psnr(out, ref));
}

static int cmd_verify(const Weights& net) {
    Tensor in, ref;
    if (!read_tensor(MODEL_DIR + "ref_in.bin", in) || !read_tensor(MODEL_DIR + "ref_out.bin", ref)) {
        printf("missing reference tensors, run train/export.py\n");
        return 1;
    }
    printf("comparing against PyTorch output, input %dx%d\n\n", in.h, in.w);

    check("cpu_naive", run_naive(net, in), ref);
    check("cpu_accelerate", run_accel(net, in), ref);

    MPSNet* mps = mps_create(net);
    check("mpsgraph_gpu", mps_run(mps, in), ref);
    mps_destroy(mps);

    for (int bits : {32, 16}) {
        std::string compiled = coreml_compile(model_path(in.h, in.w, bits), nullptr);
        if (compiled.empty()) continue;
        for (int units = 0; units < 4; units++) {
            CoreMLModel* m = coreml_load(compiled, units, nullptr);
            if (!m) continue;
            Tensor out;
            std::string name = "coreml_fp" + std::to_string(bits) + "_" + units_name(units);
            if (coreml_run(m, in, out)) check(name.c_str(), out, ref);
            coreml_destroy(m);
        }
    }

    // no PyTorch reference at the bigger sizes so compare to the Accelerate output instead
    for (int i = 1; i < 4; i++) {
        int h = SIZES[i][0], w = SIZES[i][1];
        Tensor big = randInput(h, w);
        Tensor expect = run_accel(net, big);
        printf("\ncomparing against cpu_accelerate, input %dx%d\n\n", w, h);

        mps = mps_create(net);
        check("mpsgraph_gpu", mps_run(mps, big), expect);
        mps_destroy(mps);

        for (int bits : {32, 16}) {
            std::string compiled = coreml_compile(model_path(h, w, bits), nullptr);
            if (compiled.empty()) continue;
            for (int units = 0; units < 4; units++) {
                CoreMLModel* m = coreml_load(compiled, units, nullptr);
                if (!m) continue;
                Tensor out;
                std::string name = "coreml_fp" + std::to_string(bits) + "_" + units_name(units);
                if (coreml_run(m, big, out)) check(name.c_str(), out, expect);
                coreml_destroy(m);
            }
        }
    }
    return 0;
}

static void usage() {
    printf("usage:\n");
    printf("  upscale-bench verify\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string cmd = argv[1];

    Weights net;
    if (!load_weights(MODEL_DIR + "weights.bin", net)) {
        printf("could not load models/weights.bin, run train/export.py first\n");
        return 1;
    }

    if (cmd == "verify") return cmd_verify(net);
    usage();
    return 1;
}
