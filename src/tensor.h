#pragma once
#include <vector>

struct Tensor {
    int c = 0, h = 0, w = 0;
    std::vector<float> data;

    Tensor() {}
    Tensor(int c_, int h_, int w_) : c(c_), h(h_), w(w_), data((size_t)c_ * h_ * w_, 0.0f) {}

    float& at(int ch, int y, int x) { return data[((size_t)ch * h + y) * w + x]; }
    float at(int ch, int y, int x) const { return data[((size_t)ch * h + y) * w + x]; }
};
