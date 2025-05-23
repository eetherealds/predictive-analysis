# Laporan Proyek Machine Learning - Dearni Lambardo Saragih

## Domain Proyek

Mengingat banyaknya orang yang minum susu, kualitas susu merupakan faktor penting dalam industri pangan dan kesehatan. Namun, banyak parameter fisik dan kimia, termasuk pH, suhu, rasa, bau, kandungan lemak, kekeruhan, dan warna memengaruhi kualitas susu. Untuk mendukung proses inspeksi kualitas di industri, sistem prediksi kualitas susu yang tepat dan efektif diperlukan.

**Rubrik/Kriteria Tambahan (Opsional)**:

Dalam proyek ini, digunakan pendekatan machine learning untuk memprediksi kualitas susu berdasarkan atribut yang sudah disebutkan di atas. Dataset yang digunakan terdiri dari 1.059 sampel susu dalam kategori kualitas rendah, sedang, dan tinggi. Tujuan dari proyek ini adalah untuk mengembangkan sistem prediksi kualitas susu berbasis machine learning dengan menggunakan algoritma Random Forest dan Support Vector Machine (SVM).

**Referensi Jurnal:**
https://www.researchgate.net/publication/376064637_Milk_Quality_Prediction_Using_Machine_Learning

## Business Understanding
### Problem Statements

- Bagaimana cara memprediksi kualitas susu berdasarkan parameter fisik dan kimia yang ada dalam dataset?
- Algoritma machine learning apa yang paling efektif dalam memprediksi kualitas susu?

### Goals

- Mengembangkan model machine learning untuk memprediksi kualitas susu berdasarkan parameter dataset.
- Membandingkan performa algoritma Random Forest dan Support Vector Machine (SVM) untuk prediksi kualitas susu.

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

**Memahami Data:**
1. Membaca data dari file csv dan menampilkan 5 baris pertama.
![image](https://github.com/user-attachments/assets/51fcb29c-cde2-4dc8-a399-4a0f659501c8)

2. Menampilkan jumlah baris dan kolom dalam DataFrame.
![image](https://github.com/user-attachments/assets/136dd4ec-8f9c-4ae3-9324-d1c3005d73a7)

3. Menampilkan informasi ringkas tentang DataFrame.
![image](https://github.com/user-attachments/assets/28668d0e-0572-4b8c-a9fc-ea8524452679)

4. Menampilkan statistik deskriptif untuk kolom numerik.
![image](https://github.com/user-attachments/assets/0d478c4f-9400-42cd-8120-b27ef7144bce)

5. Mengecek apakah terdapat nilai kosong pada setiap kolom.
![image](https://github.com/user-attachments/assets/80843e48-0219-4fcb-a26f-850c759871a7)

6. Setelah itu kolom 'Grade' diubah menjadi nilai numerik menggunakan mapping.
![image](https://github.com/user-attachments/assets/c45aa0f7-d802-482b-9625-1380663ce55a)

7. Membuat plot distribusi kelas susu.
![image](https://github.com/user-attachments/assets/06eff0d0-d525-43ca-aaf1-2668131f271b)

**Keterangan:**
- Low (0) berjumlah 429 sampel, menjadi kelas dengan jumlah data terbanyak.
- Medium (1) berjumlah 374 sampel.
- High (2) berjumlah 256 sampel, menjadi kelas dengan jumlah data paling sedikit.

**Exploratory Data Analysis**
1. Membuat grafik distribusi pH dan Temperature.
![image](https://github.com/user-attachments/assets/2a92e1d6-a8db-4da4-8db2-1797620ac2f3)

2. Membuat grafik distribusi pH dan Temperature terhadap kategori Grade.
![image](https://github.com/user-attachments/assets/92472f18-3ba5-4b64-b2dc-7ced2d30a25d)

3. Menampilkan grafik semua parameter terhadap grade.
![image](https://github.com/user-attachments/assets/ee40fcc0-7b74-445b-b3c2-bc08ab1df2bb)

4. Korelasi numerik menggunakan heatmap.
![image](https://github.com/user-attachments/assets/b8f62c3b-7ab3-412c-9f72-62166ffbb0a1)

## Data Preparation
1. Memisahkan fitur (x) dan label (y)
   - fitur (x) mencakup semua kolom kecuali 'Grade', sedangkan label (y) adalah kolom Grade.
   - Tujuan: Memisahkan variabel independen dan dependen untuk pemodelan
2. Menentukan kolom kategorikal dan numerik dari data fitur (x), lalu menampilkan hasil identifikasi kolom.
3. Membuat pipeline untuk transformasi data numerik dan kategorikal.
   - Pipeline numerik menggunakan `StandardScaler` untuk menstandarkan data numerik sehingga memiliki mean 0 dan standar deviasi 1.
   - Pipeline kategorikal menggunakan `OneHotEncoder` untuk mengubah data kategorikal menjadi representasi numerik.
4. Menggabungkan pipeline numerik dan kategorikal ke dalam satu perpocessor untuk memastikan transformasi dilakukan secara paralel pada kolom yang sesuai
5. Membagi dataset ke dalam data latih (80%) dan data uji (20%) untuk memastikan model dilatih dan diuji pada data yang berbeda untuk menghindari overfitting.
![image](https://github.com/user-attachments/assets/bad2ce77-c045-40db-9196-d561a803f4a0)

## Modeling
Dalam proyek ini saya menggunakan dua mosel baseline yaitu Random Forest dan Support Vector Machine (SVM). Random forest menghasilkan akurasi 99.53% dan F1-score 99.53%, sedangkan SVM 

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
