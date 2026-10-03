import random
import shutil
from pathlib import Path
import argparse

CLASSES = ["sampah", "lantai_bersih", "kotoran"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    parser = argparse.ArgumentParser(description="Split dataset raw menjadi train/val")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true",
                        help="Copy file. Default juga copy; flag ini dipertahankan untuk kompatibilitas.")
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("val_ratio harus antara 0 dan 1.")

    random.seed(args.seed)

    raw_root = Path("dataset/dataset_raw")
    train_root = Path("dataset/train")
    val_root = Path("dataset/val")

    total = 0
    for cls in CLASSES:
        src = raw_root / cls
        if not src.exists():
            print(f"[WARNING] Folder tidak ada: {src}")
            continue

        images = [p for p in src.iterdir() if p.suffix.lower() in VALID_EXT]
        random.shuffle(images)

        n_val = max(1, int(len(images) * args.val_ratio))
        val_images = images[:n_val]
        train_images = images[n_val:]

        (train_root / cls).mkdir(parents=True, exist_ok=True)
        (val_root / cls).mkdir(parents=True, exist_ok=True)

        # Bersihkan tujuan agar tidak terjadi duplikasi saat split ulang
        for p in (train_root / cls).glob("*"):
            if p.is_file():
                p.unlink()
        for p in (val_root / cls).glob("*"):
            if p.is_file():
                p.unlink()

        for p in train_images:
            shutil.copy2(p, train_root / cls / p.name)
        for p in val_images:
            shutil.copy2(p, val_root / cls / p.name)

        total += len(images)
        print(f"{cls:16s}: total={len(images):4d} | train={len(train_images):4d} | val={len(val_images):4d}")

    print(f"\nTotal gambar: {total}")
    print("Split selesai. Pastikan data train dan val berasal dari sesi/kondisi yang tidak identik untuk mengurangi data leakage.")

if __name__ == "__main__":
    main()
