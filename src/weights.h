#pragma once
#include <string>
#include <vector>
#include "tensor.h"

struct ConvLayer {
    int out_c, in_c, k;
    std::vector<float> w;
    std::vector<float> b;
};

struct Weights {
    std::vector<ConvLayer> layers;
    int scale = 2;
};

bool load_weights(const std::string& path, Weights& out);
bool read_tensor(const std::string& path, Tensor& out);
