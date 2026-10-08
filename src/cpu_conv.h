#pragma once
#include "tensor.h"
#include "weights.h"

Tensor conv2d_naive(const Tensor& in, const ConvLayer& l, bool relu);
Tensor pixel_shuffle(const Tensor& in, int r);
Tensor run_naive(const Weights& net, const Tensor& in);
