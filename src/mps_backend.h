#pragma once
#include <string>
#include "tensor.h"
#include "weights.h"

struct MPSNet;

MPSNet* mps_create(const Weights& net);
Tensor mps_run(MPSNet* m, const Tensor& in);
std::string mps_device_name(MPSNet* m);
void mps_destroy(MPSNet* m);
