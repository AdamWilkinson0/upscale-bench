import struct
import numpy as np
import torch
import coremltools as ct
from model import SRNet
from data import load_pairs

SIZES = [(256, 256), (512, 512), (1024, 1024), (1080, 1920)]
OUT = "../models/"

model = SRNet()
model.load_state_dict(torch.load(OUT + "srnet.pt", map_location="cpu"))
model.eval()

with open(OUT + "weights.bin", "wb") as f:
    convs = [model.conv1, model.conv2, model.conv3]
    f.write(struct.pack("ii", len(convs), 2))
    for conv in convs:
        w = conv.weight.detach().numpy().astype(np.float32)
        b = conv.bias.detach().numpy().astype(np.float32)
        f.write(struct.pack("iii", w.shape[0], w.shape[1], w.shape[2]))
        f.write(w.tobytes())
        f.write(b.tobytes())


def save_tensor(path, t):
    a = t.detach().numpy().astype(np.float32)[0]
    with open(path, "wb") as f:
        f.write(struct.pack("iii", *a.shape))
        f.write(a.tobytes())


# reference input/output for the c++ verify command
lr, _ = load_pairs("../data/BSDS300/images/test")[0]
x = lr[:, :, :256, :256].contiguous()
if x.shape[2] < 256 or x.shape[3] < 256:
    x = torch.nn.functional.pad(x, (0, 256 - x.shape[3], 0, 256 - x.shape[2]), mode="reflect")
with torch.no_grad():
    y = model(x)
save_tensor(OUT + "ref_in.bin", x)
save_tensor(OUT + "ref_out.bin", y)

for h, w in SIZES:
    example = torch.rand(1, 1, h, w)
    traced = torch.jit.trace(model, example)
    for bits, precision in [(32, ct.precision.FLOAT32), (16, ct.precision.FLOAT16)]:
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(1, 1, h, w), dtype=np.float32)],
            outputs=[ct.TensorType(name="y", dtype=np.float32)],
            compute_precision=precision,
            minimum_deployment_target=ct.target.macOS14,
            # compute_units=ct.ComputeUnit.ALL,
        )
        ml.save(f"{OUT}sr_{h}x{w}_fp{bits}.mlpackage")
        print("saved", h, w, bits)
