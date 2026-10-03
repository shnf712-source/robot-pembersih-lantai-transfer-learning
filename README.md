# Robot Pembersih Lantai — Transfer Learning

Proyek ini merupakan implementasi **transfer learning untuk klasifikasi kondisi lantai** pada konsep robot pembersih lantai.

Model digunakan untuk mengklasifikasikan gambar menjadi tiga kelas:

* `sampah`
* `lantai_bersih`
* `kotoran`

## 1. Tujuan

Membangun model computer vision yang dapat membantu robot pembersih lantai mengenali kondisi lingkungan berdasarkan citra kamera.

Metode yang digunakan adalah **transfer learning** menggunakan model **ResNet-18 pretrained**, dengan pendekatan feature extraction.

## 2. Dataset

Dataset terdiri dari tiga kelas:

| Kelas         |   Total |   Train | Validation |
| ------------- | ------: | ------: | ---------: |
| Sampah        |      60 |      48 |         12 |
| Lantai Bersih |      50 |      40 |         10 |
| Kotoran       |      50 |      40 |         10 |
| **Total**     | **160** | **128** |     **32** |

Pembagian dataset menggunakan rasio:

* 80% training
* 20% validation
* Random seed: 42

Dataset raw disimpan pada:

```text
dataset/dataset_raw/
```

Dataset hasil pembagian disimpan pada:

```text
dataset/train/
dataset/val/
```

## 3. Metode Transfer Learning

Model yang digunakan:

**ResNet-18**

Pendekatan:

**Feature Extraction**

Pada pendekatan ini, backbone ResNet-18 pretrained digunakan untuk mengekstraksi fitur gambar, kemudian bagian klasifikasi digunakan untuk membedakan tiga kelas target.

Input model:

```text
224 × 224 RGB
```

Kelas output:

```text
kotoran
lantai_bersih
sampah
```

## 4. Struktur Proyek

```text
robot-pembersih-lantai-transfer-learning/
│
├── dataset/
│   ├── dataset_raw/
│   │   ├── sampah/
│   │   ├── lantai_bersih/
│   │   └── kotoran/
│   ├── train/
│   └── val/
│
├── models/
│   └── resnet18_feature.pth
│
├── results/
│   ├── history_feature.csv
│   ├── comparison.csv
│   └── grafik hasil eksperimen
│
├── RecycleNet/
├── TrashNet/
│
├── capture.py
├── split.py
├── train.py
├── evaluate.py
├── latency.py
├── plot_results.py
├── EXPERIMENT_CHECKLIST.md
├── requirements.txt
└── README.md
```

## 5. Training

Training dilakukan menggunakan:

```bash
python train.py
```

Training menggunakan 10 epoch.

Model terbaik disimpan pada:

```text
models/resnet18_feature.pth
```

Hasil training terbaik:

```text
Best validation accuracy: 96.88%
```

## 6. Evaluasi Model

Evaluasi dilakukan menggunakan:

```bash
python evaluate.py --mode feature
```

Hasil:

```text
Accuracy: 96.88%
```

Confusion matrix:

```text
Classes: ['kotoran', 'lantai_bersih', 'sampah']

[[9, 0, 1],
 [0, 10, 0],
 [0, 0, 12]]
```

Interpretasi:

* `kotoran`: 9 dari 10 gambar berhasil diklasifikasikan dengan benar.
* `lantai_bersih`: 10 dari 10 gambar berhasil diklasifikasikan dengan benar.
* `sampah`: 12 dari 12 gambar berhasil diklasifikasikan dengan benar.
* Terdapat 1 gambar `kotoran` yang diprediksi sebagai `sampah`.

## 7. Hasil Performa

| Parameter           |                        Hasil |
| ------------------- | ---------------------------: |
| Model               | ResNet-18 Feature Extraction |
| Validation Accuracy |                   **96.88%** |
| Input               |                224 × 224 RGB |
| Device              |                          CPU |
| Latency             |           **98.17 ms/frame** |
| FPS                 |                **10.19 FPS** |
| Training Epoch      |                           10 |

Pengukuran latency dilakukan menggunakan:

```bash
python latency.py
```

## 8. Visualisasi Hasil

Grafik hasil eksperimen dibuat menggunakan:

```bash
python plot_results.py
```

Hasil visualisasi disimpan pada folder:

```text
results/
```

## 9. Cara Menjalankan Proyek

Install dependency:

```bash
pip install -r requirements.txt
```

Membagi dataset:

```bash
python split.py --copy
```

Training model:

```bash
python train.py
```

Evaluasi:

```bash
python evaluate.py --mode feature
```

Membuat grafik:

```bash
python plot_results.py
```

Mengukur latency:

```bash
python latency.py
```

## 10. Kesimpulan

Model **ResNet-18 dengan pendekatan feature extraction** berhasil digunakan untuk klasifikasi kondisi lantai dengan tiga kelas, yaitu `sampah`, `lantai_bersih`, dan `kotoran`.

Pada dataset validation sebanyak 32 gambar, model memperoleh **accuracy 96.88%**.

Pengujian performa pada CPU menghasilkan latency **98.17 ms/frame** atau sekitar **10.19 FPS**.

Hasil tersebut menunjukkan bahwa model dapat digunakan sebagai bagian dari sistem persepsi pada konsep robot pembersih lantai, dengan catatan bahwa performa pada lingkungan nyata perlu diuji menggunakan variasi pencahayaan, posisi objek, background, occlusion, dan kondisi lingkungan lainnya.
