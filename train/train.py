import torch
import torch.nn.functional as F
from model import SRNet
from data import load_pairs, get_batch, psnr

STEPS = 20000
BATCH = 64
device = "mps" if torch.backends.mps.is_available() else "cpu"


def evaluate(model, pairs):
    model.eval()
    ours = 0
    bicubic = 0
    with torch.no_grad():
        for lr, hr in pairs:
            out = model(lr.to(device)).cpu().clamp(0, 1)
            up = F.interpolate(lr, scale_factor=2, mode="bicubic").clamp(0, 1)
            ours += psnr(out, hr)
            bicubic += psnr(up, hr)
    model.train()
    return ours / len(pairs), bicubic / len(pairs)


def main():
    torch.manual_seed(0)
    train_pairs = load_pairs("../data/BSDS300/images/train")
    test_pairs = load_pairs("../data/BSDS300/images/test")
    print(len(train_pairs), "train images,", len(test_pairs), "test images")

    model = SRNet().to(device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8000, gamma=0.3)

    for step in range(1, STEPS + 1):
        x, target = get_batch(train_pairs, BATCH)
        x, target = x.to(device), target.to(device)
        loss = F.mse_loss(model(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0:
            ours, bic = evaluate(model, test_pairs)
            print(f"step {step} loss {loss.item():.5f} test psnr {ours:.2f} dB (bicubic {bic:.2f} dB)")

    torch.save(model.state_dict(), "../models/srnet.pt")


if __name__ == "__main__":
    main()
