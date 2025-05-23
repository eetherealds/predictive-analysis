# Laporan Proyek Machine Learning - Dearni Lambardo Saragih

## Domain Proyek

Mengingat banyaknya orang yang minum susu, kualitas susu merupakan faktor penting dalam industri pangan dan kesehatan. Namun, banyak parameter fisik dan kimia, termasuk pH, suhu, rasa, bau, kandungan lemak, kekeruhan, dan warna memengaruhi kualitas susu. Untuk mendukung proses inspeksi kualitas di industri, sistem prediksi kualitas susu yang tepat dan efektif diperlukan.

**Rubrik/Kriteria Tambahan (Opsional)**:
Dalam proyek ini, digunakan pendekatan machine learning untuk memprediksi kualitas susu berdasarkan atribut yang sudah disebutkan di atas. Dataset yang digunakan terdiri dari 1.059 sampel susu dalam kategori kualitas rendah, sedang, dan tinggi. Tujuan dari proyek ini adalah untuk mengembangkan sistem prediksi kualitas susu berbasis machine learning dengan menggunakan algoritma Random Forest dan Support Vector Machine (SVM).

Referensi Jurnal: https://www.researchgate.net/publication/376064637_Milk_Quality_Prediction_Using_Machine_Learning

## Business Understanding
### Problem Statements

- Bag
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
    ### Solution statements
    -  Mengimplementasikan dua algoritma machine learning (Random Forest dan SVM) sebagai baseline model untuk memprediksi kualitas susu.
    -  Melakukan evaluasi model menggunakan metrik akurasi, F1-score, dan confusion matrix untuk menganalisis performa dan tingkat kesalahan prediksi pada setiap kelas.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
