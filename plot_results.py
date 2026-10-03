from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results")
modes = ["feature", "partial", "scratch"]

for mode in modes:
    csv_path = RESULTS / f"history_{mode}.csv"
    if not csv_path.exists():
        print(f"Skip {mode}: {csv_path} belum ada.")
        continue

    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"] * 100, label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"] * 100, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"ResNet-18 - {mode}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS / f"accuracy_{mode}.png", dpi=150)
    plt.close()

print("Grafik selesai dibuat di folder results/.")
