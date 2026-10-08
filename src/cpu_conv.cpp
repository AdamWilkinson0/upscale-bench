#include "cpu_conv.h"

Tensor conv2d_naive(const Tensor& in, const ConvLayer& l, bool relu) {
    Tensor out(l.out_c, in.h, in.w);
    int pad = l.k / 2;
    for (int oc = 0; oc < l.out_c; oc++) {
        for (int y = 0; y < in.h; y++) {
            for (int x = 0; x < in.w; x++) {
                float sum = l.b[oc];
                for (int ic = 0; ic < l.in_c; ic++) {
                    for (int ky = 0; ky < l.k; ky++) {
                        int iy = y + ky - pad;
                        if (iy < 0 || iy >= in.h) continue;
                        for (int kx = 0; kx < l.k; kx++) {
                            int ix = x + kx - pad;
                            if (ix < 0 || ix >= in.w) continue;
                            float wv = l.w[((oc * l.in_c + ic) * l.k + ky) * l.k + kx];
                            sum += wv * in.at(ic, iy, ix);
                        }
                    }
                }
                if(relu && sum<0) sum = 0;
                out.at(oc, y, x) = sum;
            }
        }
    }
    return out;
}

Tensor pixel_shuffle(const Tensor& in, int r) {
    int out_c = in.c / (r * r);
    Tensor out(out_c, in.h * r, in.w * r);
    for (int c = 0; c < out_c; c++) {
        for (int y = 0; y < in.h; y++) {
            for (int x = 0; x < in.w; x++) {
                for (int dy = 0; dy < r; dy++) {
                    for (int dx = 0; dx < r; dx++) {
                        out.at(c, y * r + dy, x * r + dx) = in.at(c * r * r + dy * r + dx, y, x);
                    }
                }
            }
        }
    }
    return out;
}

Tensor run_naive(const Weights& net, const Tensor& in) {
    Tensor x = in;
    int n = net.layers.size();
    for (int i = 0; i < n; i++)
        x = conv2d_naive(x, net.layers[i], i != n-1);
    return pixel_shuffle(x, net.scale);
}
