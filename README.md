# Credit Score Classification

Proyek akhir mata kuliah **Model Deployment** (DTSC6012001), BINUS University.

Model yang menggolongkan skor kredit nasabah menjadi **Good**, **Standard**, atau **Poor**, beserta seluruh jalur dari data mentah sampai model yang bisa dipakai: pipeline pelatihan, pencatatan eksperimen dengan MLflow, aplikasi prediksi Streamlit, dan deployment ke AWS SageMaker.

## Hasil

| Model | F1 macro (validasi silang per nasabah, setelah GridSearchCV) |
|---|---|
| Logistic Regression | 0,640 |
| Decision Tree | 0,656 |
| Gradient Boosting | 0,653 |
| **Random Forest** | **0,669** |

Setiap kombinasi hyperparameter dicatat di MLflow. Tabel di bawah adalah run induk tiap model, dengan skor validasi terbaik, skor data uji (hanya untuk model terpilih), dan skor data latih.

![Run MLflow untuk empat model: best_cv_f1_macro, test_f1_macro, dan train_f1_macro](docs/screenshots/mlflow-runs.png)

Random Forest terpilih, lalu diuji sekali pada data uji yang tidak pernah dilihat selama pelatihan:

| Metrik (data uji) | Random Forest | Patokan: selalu menebak kelas terbanyak |
|---|---|---|
| F1 macro | **0,689** | 0,230 |
| Accuracy | 0,693 | 0,528 |
| Recall macro | 0,737 | 0,333 |

F1 macro dipakai sebagai metrik utama karena kelasnya tidak seimbang (Standard 53%, Poor 29%, Good 18%). Accuracy saja bisa menyesatkan: menebak "Standard" untuk semua nasabah sudah memberi accuracy 53%.

## Keputusan penting

**Data dibagi per nasabah, bukan per baris.** Dataset berisi 25.000 baris, tetapi hanya dari 11.254 nasabah, karena satu nasabah dicatat setiap bulan. Kalau dibagi acak per baris, sekitar 80% baris di data uji milik nasabah yang juga ada di data latih, sehingga model dinilai pada orang yang sudah pernah dilihatnya. Karena itu pembagian data memakai `GroupShuffleSplit` dan validasi silang memakai `StratifiedGroupKFold`, keduanya berdasarkan `Customer_ID`. Dengan model dan pembersihan yang sama, validasi acak per baris memberi F1 macro 0,686, sedangkan validasi per nasabah 0,669. Selisihnya kecil, jadi model tidak sekadar menghafal nasabah, tetapi skor per nasabah itu yang jujur.

**Nilai salah input dianggap tidak diketahui, bukan dipotong.** Data berisi umur 4824, jumlah pinjaman -100, dan suku bunga 1663%. Nilai seperti ini diganti NaN lalu diisi median, alih-alih dipotong ke batas wajar. Kalau dipotong ke 90, muncul 473 nasabah yang tiba-tiba berumur tepat 90 tahun, padahal umur aslinya tidak diketahui. Skor model untuk kedua cara hampir sama (0,689 dan 0,690), jadi alasannya bukan skor, melainkan supaya data yang dipakai model tetap jujur. Batas wajar tiap kolom diambil dari celah di data: misalnya suku bunga asli ada di 1–34%, tidak ada nilai 35–72%, dan nilai rusak mulai dari 73%. Untuk umur dan cicilan bulanan (EMI) celahnya tidak bersih, sehingga sebagian kecil nilai rusak masih lolos (lihat Keterbatasan).

**Pembersihan data yang sama dipakai saat pelatihan dan prediksi.** Aplikasi Streamlit memanggil fungsi `clean()` yang sama dengan pipeline pelatihan, dan model disimpan sebagai satu `sklearn.Pipeline` (imputasi, scaling, one-hot encoding, lalu classifier). Dengan begitu tidak ada perbedaan perlakuan data antara pelatihan dan pemakaian.

## Isi repo

```
Soal1a_Notebook_EDA/
  credit_score_EDA.ipynb     eksplorasi data, pembersihan, dan perbandingan model
  data_C.csv                 dataset (25.000 baris, 28 kolom)

Soal1bc_Local_Pipeline/credit_scoring/local/
  pipeline.py                menjalankan seluruh tahap berurutan
  data_ingestion.py          1. memuat dan memvalidasi CSV
  preprocessing.py           2. membersihkan data dan membagi per nasabah
  train.py                   3. GridSearchCV untuk 4 model, setiap percobaan dicatat di MLflow
  evaluation.py              4. menguji model terbaik pada data uji
  inferencing.py             fungsi prediksi untuk satu nasabah
  app.py                     aplikasi Streamlit

Soal2_AWS_Pipeline/
  pipeline_aws.py            SageMaker Pipelines: ingestion, preprocessing, training, evaluation
  *_aws.py                   skrip untuk tiap tahap di container SageMaker
  inference_aws.py           handler model untuk SageMaker Endpoint
  deploy_endpoint.ipynb      mengunggah model ke S3 dan membuat endpoint
  app_aws.py                 aplikasi Streamlit yang memanggil endpoint

docs/screenshots/            bukti deployment di AWS
```

Pipeline lokal ditutup dengan *deployment gate*: model hanya disetujui kalau F1 macro pada data uji minimal 0,65.

## Deployment di AWS

Versi AWS memakai pembersihan data, pembagian per nasabah, dan grid hyperparameter yang sama dengan versi lokal. Seluruh alurnya dijalankan di AWS Academy Learner Lab:

**1. SageMaker Pipelines.** Empat tahap (ingest, preprocess, train, evaluate) dijalankan dalam local mode di notebook instance, dan semuanya berhasil. Random Forest kembali terpilih. Di data uji, F1 macro 0,691 dan accuracy 0,702, lolos ambang 0,65. Angkanya sedikit berbeda dari versi lokal (0,689) karena container SageMaker memakai scikit-learn 1.4.2.

![Output SageMaker Pipelines: empat tahap Succeeded dan F1 macro lolos ambang](docs/screenshots/aws-pipeline.png)

Di local mode, SDK SageMaker tidak menyalin hasil tiap tahap ke folder tujuan. Karena itu `pipeline_aws.py` menjalankan ulang skrip yang sama langsung di notebook instance, dan file model yang di-deploy diambil dari hasil itu.

**2. SageMaker Endpoint.** Model dikemas, diunggah ke S3, lalu di-deploy di instance `ml.m5.large`. Smoke test mengirim tiga nasabah dari data uji (Good, Standard, Poor). Ketiganya ditebak benar, dan hasilnya sama dengan uji fungsi endpoint di notebook sebelum deploy.

![Endpoint credit-score-endpoint berstatus InService](docs/screenshots/aws-endpoint.png)

![Smoke test: tiga nasabah ditebak benar oleh endpoint](docs/screenshots/aws-smoke-test.png)

**3. Aplikasi Streamlit di EC2.** `app_aws.py` dijalankan di instance EC2 `t3.small` dengan IAM role yang boleh memanggil endpoint. Form dikirim ke endpoint lewat boto3, dan hasilnya ditampilkan di halaman yang dibuka dari IP publik instance.

![Aplikasi di EC2: nasabah dengan profil rata-rata digolongkan Standard](docs/screenshots/aws-app-standard.png)

![Aplikasi di EC2: nasabah dengan banyak pinjaman dan Credit Mix Bad digolongkan Poor](docs/screenshots/aws-app-poor.png)

![Instance EC2 credit-score-app dengan IP publik yang sama dengan alamat aplikasi](docs/screenshots/aws-ec2-instance.png)

Setelah pengujian selesai, instance EC2 di-terminate dan endpoint dihapus supaya tidak terus menimbulkan biaya. Karena itu `app_aws.py` tidak bisa dicoba tanpa membuat endpoint baru dengan `deploy_endpoint.ipynb`.

## Keterbatasan

- Dari 1.457 nasabah Poor di data uji, 167 (11%) digolongkan Good. Bagi pemberi kredit, ini kesalahan yang paling mahal, karena nasabah berisiko tinggi bisa lolos.
- Recall Standard hanya 0,60. Banyak nasabah Standard digeser ke Good (488) atau Poor (569), kemungkinan karena `class_weight="balanced"` menaikkan bobot kelas yang lebih kecil.
- F1 di data latih 0,84, sedangkan di data uji 0,69, jadi model masih cenderung menghafal data latih.
- Pengaturan terbaik berada di ujung grid (`max_depth` 16, 200 pohon). Kedalaman 20, 24, dan tanpa batas sudah diuji terpisah di notebook EDA bagian 6b: CV hanya naik paling banyak 0,001, jauh di bawah simpangan antar-fold (sekitar 0,007), sementara ukuran model naik dua kali lipat.
- Batas wajar belum menangkap semua salah input: satu umur 95 tahun (nasabah yang sama berumur 24 di bulan lain) dan 91 nilai EMI rusak di kisaran 1.000–5.000 masih lolos. Pengecekan konsistensi per nasabah antarbulan bisa menangkapnya.
- Gradient Boosting tidak punya pengaturan `class_weight`, jadi perbandingannya dengan tiga model lain tidak sepenuhnya setara.
- Setiap baris diperlakukan terpisah. Model belum memanfaatkan perubahan kondisi nasabah dari bulan ke bulan.

## Pelajaran dari proyek ini

- Periksa satuan data sebelum membaginya. Satu nasabah muncul di beberapa baris, jadi data dibagi per nasabah. Pengaruhnya diukur langsung (0,686 dengan pembagian acak, 0,669 per nasabah), bukan diasumsikan.
- Tidak semua keputusan pembersihan mengubah skor. Mengganti nilai salah input dengan NaN atau memotongnya memberi skor yang hampir sama. Alasan memilih NaN adalah supaya data tetap jujur, dan alasan itu perlu disampaikan apa adanya.
- Deployment sering gagal di detail kecil: kode AWS sempat berbeda dari versi lokal, satu file tertulis dua kali sampai tidak bisa dijalankan, dan smoke test mengirim kategori sebagai angka. Skrip cloud diuji dulu di laptop, dan smoke test memeriksa bahwa hasil endpoint sama dengan hasil lokal.
- Resource cloud harus dihapus setelah dipakai. Endpoint dan instance EC2 dihapus berdasarkan namanya, supaya pembersihan tidak bergantung pada kondisi notebook.

## Menjalankan secara lokal

Siapkan environment dengan Python 3.12:

```
cd Soal1bc_Local_Pipeline/credit_scoring/local
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Melatih dan menguji model:

```
python pipeline.py
```

Membuka aplikasi prediksi:

```
streamlit run app.py
```

Melihat seluruh percobaan di MLflow (setelah `pipeline.py` dijalankan):

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

![Aplikasi Streamlit lokal: form data nasabah dan hasil Credit Score: Good](docs/screenshots/lokal-app-good.png)

Model final (`artifacts/best_model.pkl`) ikut disimpan di repo, jadi aplikasi bisa langsung dijalankan tanpa melatih ulang. File ini disimpan terkompresi: 14 MB, dibanding 64 MB tanpa kompresi, dengan prediksi yang sama persis. Menjalankan `python pipeline.py` akan membuatnya ulang.

## Data

Dataset C yang diberikan untuk tugas ini, berasal dari dataset publik [Credit score classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) di Kaggle (lisensi CC0: Public Domain). Berisi data keuangan dan perilaku pembayaran nasabah per bulan, dengan banyak nilai kotor dan salah input.
