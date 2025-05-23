# Laporan Proyek Machine Learning - Dearni Lambardo Saragih

## Domain Proyek

Mengingat banyaknya orang yang minum susu, kualitas susu merupakan faktor penting dalam industri pangan dan kesehatan. Namun, banyak parameter fisik dan kimia, termasuk pH, suhu, rasa, bau, kandungan lemak, kekeruhan, dan warna memengaruhi kualitas susu. Untuk mendukung proses inspeksi kualitas di industri, sistem prediksi kualitas susu yang tepat dan efektif diperlukan.

**Rubrik/Kriteria Tambahan (Opsional)**:

Dalam proyek ini, digunakan pendekatan machine learning untuk memprediksi kualitas susu berdasarkan atribut yang sudah disebutkan di atas. Dataset yang digunakan terdiri dari 1.059 sampel susu dalam kategori kualitas rendah, sedang, dan tinggi. Tujuan dari proyek ini adalah untuk mengembangkan sistem prediksi kualitas susu berbasis machine learning dengan menggunakan algoritma Random Forest dan Support Vector Machine (SVM).

Referensi Jurnal: https://www.researchgate.net/publication/376064637_Milk_Quality_Prediction_Using_Machine_Learning

## Business Understanding
### Problem Statements

- Bagaimana cara memprediksi kualitas susu berdasarkan parameter fisik dan kimia yang ada dalam dataset?
- Algoritma machine learning apa yang paling efektif dalam memprediksi kualitas susu?

### Goals

- Mengembangkan model machine learning untuk memprediksi kualitas susu berdasarkan parameter dataset.
- Membandingkan performa algoritma Random Forest dan Support Vector Machine (SVM) untuk prediksi kualitas susu.

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution statements
- Mengimplementasikan dua algoritma machine learning (Random Forest dan SVM) sebagai baseline model untuk memprediksi kualitas susu.
- Melakukan evaluasi model menggunakan metrik akurasi, F1-score, dan confusion matrix untuk menganalisis performa dan tingkat kesalahan prediksi pada setiap kelas.

## Data Understanding
Dataset yang digunakan dalam proyek ini memiliki 1.059 sampel susu dengan 8 atribut yang mencakup parameter fisik dan kimia. Dataset ini berisi kategori kualitas susu yang diklasifikasikan sebagai rendah (low), sedang (medium), dan tinggi (high) 

Sumber dataset: https://www.kaggle.com/datasets/cpluzshrijayan/milkquality/data

### Variabel-variabel pada Milk Quality dataset adalah sebagai berikut:
- pH : Tingkat keasaman susu
- Temprature : Suhu susu saat pengukuran
- Taste : Indikator rasa (0 = tidak enak, 1 = enak)
- Odor : Indikator bau (0 = bau tidak sedap, 1 = normal)
- Fat : Kandungan lemak (0 = rendah, 1 = tinggi)
- Turbidity : Kekeruhan (0 = rendah, 1 = tinggi)
- Colour : Nilai intensitas warna susu
- Grade : Kualitas susu (low, medium, high)

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
