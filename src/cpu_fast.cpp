#include "cpu_fast.h"
#include "cpu_conv.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cstring>

// full im2col matrix for 1080p is several GB, so do a strip of rows at a time
// TODO try 16 or 64 and see if it changes anything
static const int ROWS_PER_STRIP = 32;

static void im2colStrip(const Tensor& in, int k, int y0, int rows, std::vector<float>& col) {
    int pad = k / 2;
    int w = in.w;
    size_t idx = 0;
    for (int ic = 0; ic < in.c; ic++) {
        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                for (int y = y0; y < y0 + rows; y++) {
                    int iy = y + ky - pad;
                    float* dst = &col[idx];
                    idx += w;
                    if (iy < 0 || iy >= in.h) {
                        memset(dst, 0, w * sizeof(float));
                        continue;
                    }
                    const float* src = &in.data[((size_t)ic * in.h + iy) * w];
                    for (int x = 0; x < w; x++) {
                        int ix = x + kx - pad;
                        dst[x] = (ix < 0 || ix >= w) ? 0.0f : src[ix];
                    }
                }
            }
        }
    }
}

Tensor conv2d_gemm(const Tensor& in, const ConvLayer& l, bool relu) {
    Tensor out(l.out_c, in.h, in.w);
    int patch = l.in_c * l.k * l.k;
    std::vector<float> col((size_t)patch * ROWS_PER_STRIP * in.w);
    std::vector<float> tmp((size_t)l.out_c*ROWS_PER_STRIP*in.w);

    for (int y0 = 0; y0 < in.h; y0 += ROWS_PER_STRIP) {
        int rows = std::min(ROWS_PER_STRIP, in.h - y0);
        int n = rows * in.w;
        im2colStrip(in, l.k, y0, rows, col);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l.out_c, n, patch,
                    1.0f, l.w.data(), patch, col.data(), n, 0.0f, tmp.data(), n);
        //printf("strip %d\n", y0);
        for (int oc = 0; oc < l.out_c; oc++) {
            float *dst = &out.data[((size_t)oc * in.h + y0) * in.w];
            const float *src = &tmp[(size_t)oc * n];
            float b = l.b[oc];
            for(int i=0;i<n;i++){
                float v = src[i] + b;
                if (relu && v < 0) v = 0;
                dst[i] = v;
            }
        }
    }
    return out;
}

Tensor run_accel(const Weights& net, const Tensor& in) {
    Tensor x = in;
    for (size_t i = 0; i < net.layers.size(); i++) {
        bool last = (i == net.layers.size() - 1);
        x = conv2d_gemm(x, net.layers[i], !last);
    }
    return pixel_shuffle(x, net.scale);
}
