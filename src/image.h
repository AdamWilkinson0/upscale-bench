#pragma once
#include <string>
#include "tensor.h"

// RGB image as a 3 channel tensor with values 0..1
bool load_image(const std::string& path, Tensor& rgb);
bool save_image(const std::string& path, const Tensor& rgb);

void rgb_to_ycbcr(const Tensor& rgb, Tensor& y, Tensor& cb, Tensor& cr);
Tensor ycbcr_to_rgb(const Tensor& y, const Tensor& cb, const Tensor& cr);
Tensor bilinear_up(const Tensor& in, int scale);
