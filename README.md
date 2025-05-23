# Laporan Proyek Machine Learning - Dearni Lambardo Saragih

## Domain Proyek: Pangan dan Kesehatan

Mengingat banyaknya orang yang minum susu, kualitas susu merupakan faktor penting dalam industri pangan dan kesehatan. Namun, banyak parameter fisik dan kimia, termasuk pH, suhu, rasa, bau, kandungan lemak, kekeruhan, dan warna memengaruhi kualitas susu. Untuk mendukung proses inspeksi kualitas di industri, sistem prediksi kualitas susu yang tepat dan efektif diperlukan.

Dalam proyek ini, digunakan pendekatan machine learning untuk memprediksi kualitas susu berdasarkan atribut yang sudah disebutkan di atas. Dataset yang digunakan terdiri dari 1.059 sampel susu dalam kategori kualitas rendah, sedang, dan tinggi. Tujuan dari proyek ini adalah untuk mengembangkan sistem prediksi kualitas susu berbasis machine learning dengan menggunakan algoritma Random Forest dan Support Vector Machine (SVM).

[Link Jurnal](https://www.researchgate.net/publication/376064637_Milk_Quality_Prediction_Using_Machine_Learning)

## Business Understanding
### Problem Statements

- Bagaimana cara memprediksi kualitas susu berdasarkan parameter fisik dan kimia yang ada dalam dataset?
- Algoritma machine learning apa yang paling efektif dalam memprediksi kualitas susu?

### Goals

- Mengetahui cara memprediksi kualitas susu berdasarkan parameter fisik dan kimia yang ada dalam dataset.
- Mengetahui algoritma machine learning yang paling efektif dalam memprediksi kualitas susu.
  
### Solution statements

- Mengimplementasikan dua algoritma machine learning (Random Forest dan SVM) sebagai baseline model untuk memprediksi kualitas susu.
- Melakukan evaluasi model menggunakan metrik akurasi, F1-score, dan confusion matrix untuk menganalisis performa dan tingkat kesalahan prediksi paling efektif dalam memprediksi kualitas susu pada setiap kelas

## Data Understanding
Dataset yang digunakan dalam proyek ini memiliki 1.059 sampel susu dengan 8 atribut yang mencakup parameter fisik dan kimia. Dataset ini berisi kategori kualitas susu yang diklasifikasikan sebagai rendah (low), sedang (medium), dan tinggi (high) 

Sumber dataset: https://www.kaggle.com/datasets/cpluzshrijayan/milkquality/data

### Variabel pada Milk Quality dataset
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

2. Menampilkan jumlah baris dan kolom dalam DataFrame. Disini dataset memiliki 1059 baris dan 8 kolom.
   
![image](https://github.com/user-attachments/assets/136dd4ec-8f9c-4ae3-9324-d1c3005d73a7)

3. Menampilkan informasi ringkas tentang DataFrame. Disini semua kolom memiliki data yang lengkap (tidak terdapat missing values).
   
![image](https://github.com/user-attachments/assets/28668d0e-0572-4b8c-a9fc-ea8524452679)

4. Menampilkan statistik deskriptif untuk kolom numerik.
   
![image](https://github.com/user-attachments/assets/0d478c4f-9400-42cd-8120-b27ef7144bce)

**Keterangan:**
- count : Jumlah data (1059 untuk semua kolom)
- mean : Rata-rata (contoh: pH ~6.63, Temperatur ~44.2°C)
- std : Simpangan baku (variabilitas data)
- min : Nilai minimum (misal pH = 3.0)
- 25% : Kuartil pertama (batas 25% data terbawah)
- 50% : Median (kuartil kedua / tengah)
- 75% : Kuartil ketiga (batas 25% data teratas)
- max : Nilai maksimum
  
5. Mengecek apakah terdapat nilai kosong pada setiap kolom. Disini tidak terdapat nilai kosong di seluruh kolom dataset.
   
![image](https://github.com/user-attachments/assets/80843e48-0219-4fcb-a26f-850c759871a7)

6. Lalu kita mengubah kolom `Grade` menjadi nilai numerik menggunakan mapping. Sekarang diasumsikan Low (0), Meidum (1), dan High (2).
    
![image](https://github.com/user-attachments/assets/c45aa0f7-d802-482b-9625-1380663ce55a)

7. Membuat plot distribusi kelas susu.
    
![image](https://github.com/user-attachments/assets/06eff0d0-d525-43ca-aaf1-2668131f271b)

**Keterangan:**
- Low (0) berjumlah 429 sampel, menjadi kelas dengan jumlah data terbanyak.
- Medium (1) berjumlah 374 sampel.
- High (2) berjumlah 256 sampel, menjadi kelas dengan jumlah data paling sedikit.

**Exploratory Data Analysis (EDA)**
1. **Membuat grafik distribusi pH dan Temperature.**

![image](https://github.com/user-attachments/assets/2a92e1d6-a8db-4da4-8db2-1797620ac2f3)

**Keterangan:**

Gambar Kiri (Distribusi pH)
- Mayoritas sampel memiliki nilai pH antara 6-7, menunjukkan susu dalam kondisi normal.
- Nilai di luar rentang tersebut menunjukkan potensi susu asam atau terkontaminasi.

Gambar Kanan (Distribusi Temperature)
- Sebagian besar sampel berada pada suhu 35-45°C, mencerminkan suhu standar penyimpanan atau pengolahan susu.
- Nilai di atas 50°C menunjukkan kemungkinan pasteurisasi atau pemanasan.
  
2. **Membuat grafik distribusi pH dan Temperature terhadap kategori Grade.**

![image](https://github.com/user-attachments/assets/92472f18-3ba5-4b64-b2dc-7ced2d30a25d)

**Keterangan**

Gambar Kiri (pH by Grade):
- Grade Low (0) memiliki variasi pH yang besar (3-9) dengan median sekitar 6.5.
- Grade Medium (1) dan High (2) memiliki pH yang lebih stabil di kisaran 6-7.

Gambar Kanan (Temperature by Grade):
- Grade Low (0) menunjukkan variasi temperatur yang besar (35-70°C) dengan outlier di 90°C.
- Grade Medium (1) dan High (2) memiliki rentang temperatur lebih konsisten (35-45°C).

3. **Menampilkan grafik semua parameter terhadap grade.**

![image](https://github.com/user-attachments/assets/ee40fcc0-7b74-445b-b3c2-bc08ab1df2bb)

**Keterangan**

Taste by Grade
- Taste 1 (enak) banyak ditemukan pada Grade 0 (Low) dan Grade 2 (High).
- Taste 0 (tidak enak) lebih banyak pada Grade 1 (Medium).

Odor by Grade 
- Odor 0 (bau tidak sedap) mendominasi Grade 1 (Medium) dan Grade 0 (Low).
- Odor 1 (normal) mendominasi Grade 2 (High).

Fat by Grade
- Fat 0 (rendah) dominan di Grade 1 (Medium).
- Fat 1 (tinggi) lebih banyak muncul pada Grade 0 (Low) dan Grade 2 (High).

Turbidity by Grade
- Turbidity 0 (rendah) banyak pada Grade 1 (Medium).
- Turbidity 1 (tinggi) banyak pada Grade 0 (Low) dan Grade 2 (High).

Colour by Grade
- Colour 255 mendominasi semua grade, tapi Grade 0 (Low) paling banyak.
- Nilai warna di bawah 255 cenderung meningkat untuk Grade 2 (High).

4. **Korelasi numerik menggunakan heatmap.**

![image](https://github.com/user-attachments/assets/b8f62c3b-7ab3-412c-9f72-62166ffbb0a1)

**Keterangan:**
- Grade memiliki korelasi negatif yang lemah terhadap temperature, korelasi antar fitur lainnya juga terlihat lemah. Terlihat nilai mendekati 0 yang menunjukkan bahwa variabel-variabel tersebut relatif independen satu sama lain.
-  Odor dan Turbidity memiliki korelasi positif moderat (0.46), yang dapat menunjukkan hubungan antara bau dan tingkat kekeruhan susu.

## Data Preparation
1. Memisahkan fitur (x) dan label (y)
   - fitur (x) mencakup semua kolom kecuali `Grade`, sedangkan label (y) adalah kolom `Grade`.
   - Tujuan: Memisahkan variabel independen dan dependen untuk pemodelan
2. Menentukan kolom kategorikal dan numerik dari data fitur (x), lalu menampilkan hasil identifikasi kolom.
3. Membuat pipeline untuk transformasi data numerik dan kategorikal.
   - Pipeline numerik menggunakan `StandardScaler` untuk menstandarkan data numerik sehingga memiliki mean 0 dan standar deviasi 1.
   - Pipeline kategorikal menggunakan `OneHotEncoder` untuk mengubah data kategorikal menjadi representasi numerik.
4. Menggabungkan pipeline numerik dan kategorikal ke dalam satu perpocessor untuk memastikan transformasi dilakukan secara paralel pada kolom yang sesuai
5. Membagi dataset ke dalam data latih (80%) dan data uji (20%) untuk memastikan model dilatih dan diuji pada data yang berbeda untuk menghindari overfitting.

![image](https://github.com/user-attachments/assets/bad2ce77-c045-40db-9196-d561a803f4a0)

## Modeling
Pada tahap ini saya menggunakan dua algoritma machine learning yaitu Random Forest (RF) dan Support Vector Machine (SVM) untuk menyelesasikan permasalahan klasifikasi dalam Milk Quality Dataset yang diimplementasikan menggunakan pipeline untuk memastikan proses preprocessing dan pelatihan model dilakukan secara terintegrasi. 

**Random Forest**
RF adalah algoritma ensemble yang menggunakan banyak pohon keputusan untuk menghasilkan prediksi yang stabil dan akurat. RF memiliki kelebihan seperti ketahanan terhadap overfitting dan kemampuan menangani data dengan missing values, tetapi membutuhkan lebih banyak sumber daya komputasi.

Pipeline untuk model RF menggunakan  `RandomForestClassifier` dari library `sklearn.ensemble`. Disini menggunakan parameter seed `random_state=42`. Pipeline ini terdiri dari preprocessing data `preprocessor` dan pelatihan model. Data pelatihan (`X_train`, `y_train`) digunakan untuk melatih model dengan `fit`, sedangkan prediksi dilakukan pada data pengujian (`X_test`) menggunakan `predict`.

![image](https://github.com/user-attachments/assets/76642aae-3d1d-4434-bb6c-8272330aaa37)

![image](https://github.com/user-attachments/assets/1fd673ec-4b15-482a-bddc-c62299548743)

Sedangkan, pipeline SVM menggunakan `SVC` dari sklearn.svm. Pipeline ini juga terdiri dari preprocessing data dan pelatihan model. Disini menggunakan parameter seed juga yaitu `random_state=42`. Parameter `probability=True` digunakan untuk memungkinkan keluaran probabilitas prediksi. SVM bekerja dengan mencari hyperplane optimal untuk memisahkan kelas dalam data dan sangat efektif untuk data berdimensi tinggi. Namun, SVM sensitif terhadap noise dan membutuhkan tuning parameter seperti `C` dan `gamma`.

![image](https://github.com/user-attachments/assets/4a3102b4-34a2-4a2e-aecc-4c2af6fbb4a9)

![image](https://github.com/user-attachments/assets/4a822704-cd4b-4306-aaa4-5249c6cd2a40)

Karena Random Forest memiliki performa yang lebih baik dibanding SVM, maka dipilih sebagai model terbaik dalam proyek ini.

## Evaluation
Pada tahap evaluasi ini saya menggunakan beberapa metrik untuk mengukur performa model. yaitu `accuracy`, `precission`, `recall` dan `F1-score`. Disini saya memakai confusion matrix untuk memberikan visualisasi mengenai hasil klasifikasi model secara lebih rinci.

### **Metrik Evaluasi yang Digunakan**
1. Accuracy: Mengukur presentase prediksi yang benar dibandingkan dengan total data.
   \[
   Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
   \]

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
