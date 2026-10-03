import argparse
import time
import torch
import torch.nn as nn
from torchvision import models, transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["feature", "partial", "scratch"], default="feature")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(f"models/resnet18_{args.mode}.pth", map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.runs):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    ms = (elapsed / args.runs) * 1000
    fps = 1000 / ms

    print(f"Device     : {device}")
    print(f"Model      : ResNet-18 ({args.mode})")
    print(f"Input      : 224x224 RGB")
    print(f"Runs       : {args.runs}")
    print(f"Latency    : {ms:.2f} ms/frame")
    print(f"Model FPS  : {fps:.2f} FPS")

if __name__ == "__main__":
    main()
