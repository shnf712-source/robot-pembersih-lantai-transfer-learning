import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASSES = ["sampah", "lantai_bersih", "kotoran"]
IMG_SIZE = 224

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(mode, num_classes):
    if mode == "scratch":
        model = models.resnet18(weights=None)
        for p in model.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if mode == "feature":
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif mode == "partial":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_loaders(batch_size, workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder("dataset/train", transform=train_tf)
    val_ds = datasets.ImageFolder("dataset/val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, train_ds.classes

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    # Untuk feature extraction, BatchNorm backbone pretrained tetap lebih stabil
    # jika dataset kecil. Parameter backbone juga frozen.
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["feature", "partial", "scratch"], default="feature")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=0,
                        help="Windows disarankan 0; Linux bisa 2-4.")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, classes = get_loaders(args.batch_size, args.workers)
    model = build_model(args.mode, len(classes)).to(device)

    if args.lr is not None:
        optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    elif args.mode == "partial":
        optimizer = Adam([
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ])
    else:
        optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    history = []
    best_val = 0.0
    best_epoch = 0
    start_all = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        epoch_time = time.perf_counter() - t0
        history.append([epoch, train_loss, train_acc, val_loss, val_acc, epoch_time])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"{epoch_time:.1f}s"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "mode": args.mode,
                "img_size": IMG_SIZE,
                "val_acc": val_acc,
            }, f"models/resnet18_{args.mode}.pth")

    total_time = time.perf_counter() - start_all

    csv_path = Path("results") / f"history_{args.mode}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"])
        w.writerows(history)

    summary_path = Path("results") / "comparison.csv"
    rows = []
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if r["mode"] != args.mode]

    rows.append({
        "mode": args.mode,
        "best_val_accuracy": f"{best_val:.6f}",
        "best_epoch": str(best_epoch),
        "training_time_sec": f"{total_time:.2f}",
        "device": str(device),
    })

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["mode", "best_val_accuracy", "best_epoch", "training_time_sec", "device"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nModel terbaik: models/resnet18_{args.mode}.pth")
    print(f"Best validation accuracy: {best_val*100:.2f}%")
    print(f"History: {csv_path}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
