#pragma once
#include "tensor.h"
#include "weights.h"

Tensor conv2d_gemm(const Tensor& in, const ConvLayer& l, bool relu);
Tensor run_accel(const Weights& net, const Tensor& in);
