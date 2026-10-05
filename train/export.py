import struct
import numpy as np
import torch
from model import SRNet

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
