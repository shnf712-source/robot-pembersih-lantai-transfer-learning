import cv2
import csv
import argparse
from pathlib import Path
from datetime import datetime

CLASSES = ["sampah", "lantai_bersih", "kotoran"]

def main():
    parser = argparse.ArgumentParser(description="Capture dataset dari webcam")
    parser.add_argument("--class_name", required=True, choices=CLASSES,
                        help="Nama kelas: sampah, lantai_bersih, kotoran")
    parser.add_argument("--count", type=int, default=100,
                        help="Jumlah gambar yang ingin diambil")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index webcam")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Jeda minimum antar foto (detik)")
    args = parser.parse_args()

    out_dir = Path("dataset/dataset_raw") / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path("dataset/dataset_raw/metadata.csv")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak dapat dibuka. Coba --camera 1 atau cek izin kamera.")

    print("\nKontrol:")
    print("  SPACE = ambil foto")
    print("  Q     = keluar")
    print(f"\nTarget: {args.count} gambar untuk kelas '{args.class_name}'")

    captured = 0
    last_capture = 0.0

    metadata_exists = metadata_path.exists()
    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not metadata_exists:
            writer.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya", "catatan"])

        while captured < args.count:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame.")
                break

            display = frame.copy()
            cv2.putText(display, f"Kelas: {args.class_name}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(display, f"Progress: {captured}/{args.count}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(display, "SPACE=Capture | Q=Quit", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Dataset Robot Pembersih Lantai", display)
            key = cv2.waitKey(1) & 0xFF

            now = cv2.getTickCount() / cv2.getTickFrequency()
            if key == ord("q"):
                break

            if key == 32 and (now - last_capture) >= args.interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{args.class_name}_{timestamp}.jpg"
                path = out_dir / filename

                cv2.imwrite(str(path), frame)
                writer.writerow([
                    filename,
                    args.class_name,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "isi_manual: terang/redup/normal",
                    ""
                ])
                f.flush()

                captured += 1
                last_capture = now
                print(f"[{captured}/{args.count}] {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSelesai. Dataset disimpan di: {out_dir}")

if __name__ == "__main__":
    main()
