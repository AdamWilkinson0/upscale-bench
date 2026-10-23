#include "image.h"
#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

bool load_image(const std::string& path, Tensor& rgb) {
    int w, h, n;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!px) return false;
    rgb = Tensor(3, h, w);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++)
                rgb.at(c,y,x) = px[(y*w + x)*3 + c] / 255.0f;
    stbi_image_free(px);
    return true;
}

bool save_image(const std::string& path, const Tensor& rgb) {
    std::vector<unsigned char> px((size_t)rgb.h * rgb.w * 3);
    for (int y = 0; y < rgb.h; y++) {
        for (int x = 0; x < rgb.w; x++) {
            for (int c = 0; c < 3; c++) {
                float v = std::min(1.0f, std::max(0.0f, rgb.at(c, y, x)));
                px[((size_t)y * rgb.w + x) * 3 + c] = (unsigned char)std::lround(v * 255.0f);
            }
        }
    }
    return stbi_write_png(path.c_str(), rgb.w, rgb.h, 3, px.data(), rgb.w * 3) != 0;
}

void rgb_to_ycbcr(const Tensor& rgb, Tensor& y, Tensor& cb, Tensor& cr) {
    y = Tensor(1, rgb.h, rgb.w);
    cb = Tensor(1, rgb.h, rgb.w);
    cr = Tensor(1, rgb.h, rgb.w);
    for (int i = 0; i < rgb.h; i++) {
        for (int j = 0; j < rgb.w; j++) {
            float r = rgb.at(0, i, j), g = rgb.at(1, i, j), b = rgb.at(2, i, j);
            y.at(0, i, j) = 0.299f * r + 0.587f * g + 0.114f * b;
            cb.at(0, i, j) = -0.168736f * r - 0.331264f * g + 0.5f * b + 0.5f;
            cr.at(0, i, j) = 0.5f * r - 0.418688f * g - 0.081312f * b + 0.5f;
        }
    }
}

Tensor ycbcr_to_rgb(const Tensor& y, const Tensor& cb, const Tensor& cr) {
    Tensor rgb(3, y.h, y.w);
    for (int i = 0; i < y.h; i++) {
        for (int j = 0; j < y.w; j++) {
            float yy = y.at(0, i, j), b = cb.at(0, i, j) - 0.5f, r = cr.at(0, i, j) - 0.5f;
            rgb.at(0, i, j) = yy + 1.402f * r;
            rgb.at(1, i, j) = yy - 0.344136f * b - 0.714136f * r;
            rgb.at(2, i, j) = yy + 1.772f * b;
        }
    }
    return rgb;
}

// only used for the chroma channels (and as a comparison)
Tensor bilinear_up(const Tensor &in, int scale) {
    Tensor out(in.c, in.h * scale, in.w * scale);
    for (int c = 0; c < in.c; c++) {
        for (int y = 0; y < out.h; y++) {
            float sy = (y + 0.5f) / scale - 0.5f;
            int y0 = (int)std::floor(sy);
            float fy = sy - y0;
            int ya = std::clamp(y0, 0, in.h - 1), yb = std::clamp(y0 + 1, 0, in.h - 1);
            for (int x = 0; x < out.w; x++) {
                float sx = (x + 0.5f) / scale - 0.5f;
                int x0 = (int)std::floor(sx);
                float fx = sx - x0;
                int xa = std::clamp(x0, 0, in.w - 1), xb = std::clamp(x0 + 1, 0, in.w - 1);
                float top = in.at(c, ya, xa) * (1 - fx) + in.at(c, ya, xb) * fx;
                float bot = in.at(c, yb, xa) * (1 - fx) + in.at(c, yb, xb) * fx;
                out.at(c, y, x) = top * (1 - fy) + bot * fy;
            }
        }
    }
    return out;
}
