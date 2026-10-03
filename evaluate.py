import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["feature", "partial", "scratch"], required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(f"models/resnet18_{args.mode}.pth")
    ckpt = torch.load(ckpt_path, map_location=device)

    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder("dataset/val", transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    matrix = torch.zeros(len(classes), len(classes), dtype=torch.int64)

    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for true, p in zip(y, pred):
                matrix[true, p] += 1

    acc = matrix.diag().sum().item() / matrix.sum().item()
    print(f"\nMode: {args.mode}")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix")
    print("Rows = actual, columns = predicted")
    print("Classes:", classes)
    print(matrix)

if __name__ == "__main__":
    main()
