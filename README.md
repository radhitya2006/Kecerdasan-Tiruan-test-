# YieldPredict

YieldPredict adalah proyek UAS untuk memprediksi hasil panen per hektare. Aplikasi Streamlit menerima lima input, yaitu negara, tanaman, tahun, curah hujan tahunan, dan penggunaan pestisida tahunan di tingkat negara. Model Random Forest menghasilkan estimasi dalam **ton/ha**.

Proyek ini memakai Python 3.11, pandas, scikit-learn, Streamlit, dan joblib.

## Menjalankan di Windows

Buka folder `YieldPredict` di VS Code, lalu jalankan perintah berikut dari terminal PowerShell di folder proyek.

```powershell
py -3.11 -m venv .venv
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
& '.venv/Scripts/python.exe' train.py
& '.venv/Scripts/python.exe' -m streamlit run app.py
```

Buka **Local URL** yang ditampilkan Streamlit, biasanya `http://localhost:8501`. `train.py` menghasilkan `models/yield_pipeline.joblib` yang berisi preprocessing dan model. File tersebut diabaikan oleh Git, sehingga perlu dibuat lewat training saat menyiapkan proyek di komputer baru.

## Contoh input

Nilai awal formulir memakai baris pertama. Isi formulir dan klik **Prediksi hasil panen** untuk melihat hasilnya.

| Negara | Tanaman | Tahun | Curah hujan (mm) | Pestisida (ton) | Prediksi (ton/ha) |
| --- | --- | ---: | ---: | ---: | ---: |
| Indonesia | `Rice, paddy` | 2013 | 2702 | 1597 | 4,933 |
| Indonesia | `Maize` | 2012 | 2702 | 1597 | 4,109 |
| Albania | `Wheat` | 2011 | 1485 | 1302.63 | 3,838 |
| India | `Potatoes` | 2013 | 1083 | 45620 | 18,267 |
| Brazil | `Soybeans` | 2012 | 1761 | 346583 | 2,664 |
| Japan | `Rice, paddy` | 2010 | 1668 | 55576 | 6,656 |

## Dataset dan preprocessing

File mentah `data/raw/yield_df.csv` berisi 28.242 baris dari 101 negara dan 10 tanaman pada 1990–2013, dengan 2003 absen. [Crop Yield Prediction Dataset di Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

Preprocessing mengabaikan nomor urut dan `avg_temp`, lalu menyisakan 13.130 catatan unik pada tingkat negara, tanaman, dan tahun. Suhu dikeluarkan karena beberapa negara dan tahun punya lebih dari satu nilai dengan rincian pengukuran yang belum jelas. `Area` dan `Item` diubah menjadi kolom indikator dengan *one-hot encoding*. Target `hg/ha_yield` dipakai untuk pelatihan, lalu hasil prediksi dikonversi ke `ton/ha` di aplikasi.

## Evaluasi model

Konfigurasi Random Forest dengan **100 pohon** dan `random_state=42` dipilih menggunakan train 1990–2006 dan validation 2007–2009. Pipeline dilatih ulang pada 10.775 catatan gabungan sampai 2009, lalu diuji pada 2.355 catatan dari **2010–2013**.

| Model | MAE test (ton/ha) |
| --- | ---: |
| Baseline median negara–tanaman | 1,806 |
| **Random Forest** | **1,139** |

MAE Random Forest 36,9% lebih rendah dari baseline pada periode uji tersebut.

## Struktur proyek

- [`app.py`](app.py) — formulir prediksi Streamlit.
- [`train.py`](train.py) — preprocessing, pelatihan, dan penyimpanan pipeline.
- [`notebooks/01_eksplorasi_awal.ipynb`](notebooks/01_eksplorasi_awal.ipynb) — eksplorasi data, eksperimen, dan evaluasi.
- [`requirements.txt`](requirements.txt) — versi dependensi proyek.
- `data/raw/` dan `data/processed/` — lokasi data mentah dan hasil olahan jika perlu diekspor.
- `models/` — lokasi pipeline hasil training.

Untuk menjalankan ulang notebook di VS Code, instal kernel dengan `& '.venv/Scripts/python.exe' -m pip install ipykernel`, lalu pilih `.venv/Scripts/python.exe` sebagai kernel.
