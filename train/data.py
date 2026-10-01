import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_y(path):
    img = Image.open(path).convert("YCbCr")
    y = np.asarray(img, dtype=np.float32)[:, :, 0] / 255.0
    h, w = y.shape
    y = y[: h - h % 2, : w - w % 2]
    return torch.from_numpy(y)[None, None]


def downscale(hr):
    return F.interpolate(hr, scale_factor=0.5, mode="bicubic", antialias=True).clamp(0, 1)


def load_pairs(folder):
    pairs = []
    for p in sorted(glob.glob(folder + "/*.jpg")):
        hr = load_y(p)
        pairs.append((downscale(hr), hr))
    return pairs


def get_batch(pairs, batch_size, patch=32):
    lrs, hrs = [], []
    for _ in range(batch_size):
        lr, hr = pairs[np.random.randint(len(pairs))]
        h, w = lr.shape[2], lr.shape[3]
        i = np.random.randint(0, h - patch)
        j = np.random.randint(0, w - patch)
        l = lr[0, :, i:i+patch, j:j+patch]
        t = hr[0, :, 2*i:2*(i+patch), 2*j:2*(j+patch)]
        if np.random.rand() < 0.5:
            l, t = l.flip(2), t.flip(2)
        lrs.append(l)
        hrs.append(t)
    return torch.stack(lrs), torch.stack(hrs)


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 10 * np.log10(1.0 / mse)
