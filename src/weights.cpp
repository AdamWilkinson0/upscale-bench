#include "weights.h"
#include <cstdio>
#include <cstdint>

bool load_weights(const std::string& path, Weights& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t n, scale;
    fread(&n, 4, 1, f);
    fread(&scale, 4, 1, f);
    out.scale = scale;
    out.layers.clear();
    for (int i = 0; i < n; i++) {
        int32_t dims[3];
        if (fread(dims, 4, 3, f) != 3) { fclose(f); return false; }
        ConvLayer l;
        l.out_c = dims[0]; l.in_c = dims[1]; l.k = dims[2];
        l.w.resize((size_t)l.out_c * l.in_c * l.k * l.k);
        l.b.resize(l.out_c);
        fread(l.w.data(), 4, l.w.size(), f);
        fread(l.b.data(), 4, l.b.size(), f);
        out.layers.push_back(l);
    }
    fclose(f);
    return true;
}

bool read_tensor(const std::string& path, Tensor& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t dims[3];
    fread(dims, 4, 3, f);
    out = Tensor(dims[0], dims[1], dims[2]);
    size_t got = fread(out.data.data(), 4, out.data.size(), f);
    fclose(f);
    return got == out.data.size();
}
