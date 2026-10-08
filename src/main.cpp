#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_conv.h"
#include "weights.h"

static const std::string MODEL_DIR = "models/";

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

int main(int argc, char** argv) {
    Weights net;
    if (!load_weights(MODEL_DIR + "weights.bin", net)) {
        printf("could not load models/weights.bin, run train/export.py first\n");
        return 1;
    }

    Tensor in, ref;
    if (!read_tensor(MODEL_DIR + "ref_in.bin", in) || !read_tensor(MODEL_DIR + "ref_out.bin", ref)) {
        printf("missing reference tensors\n");
        return 1;
    }
    printf("verify against pytorch, input %dx%d\n\n", in.h, in.w);

    Tensor out = run_naive(net, in);
    printf("cpu_naive       max err %.2e  psnr %.1f dB\n", maxDiff(out, ref), psnr(out, ref));
    return 0;
}
