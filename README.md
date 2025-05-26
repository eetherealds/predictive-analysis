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
- Melakukan evaluasi model menggunakan metrik akurasi, F1-score, dan confusion matrix untuk menganalisis performa dan tingkat kesalahan prediksi paling efektif dalam memprediksi kualitas susu pada setiap kelas.

## Data Understanding
Dataset yang digunakan dalam proyek ini memiliki 1.059 sampel susu dengan 8 atribut yang mencakup parameter fisik dan kimia. Dataset ini berisi kategori kualitas susu yang diklasifikasikan sebagai rendah (low), sedang (medium), dan tinggi (high).

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

6. Membuat plot distribusi kelas susu.
    
![image](https://github.com/user-attachments/assets/06eff0d0-d525-43ca-aaf1-2668131f271b)

**Keterangan:**
- Low berjumlah 429 sampel, menjadi kelas dengan jumlah data terbanyak.
- Medium berjumlah 374 sampel.
- High berjumlah 256 sampel, menjadi kelas dengan jumlah data paling sedikit.

### **Exploratory Data Analysis (EDA)**
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

![image](https://github.com/user-attachments/assets/69442871-c40b-459b-b843-de550301c17d)


**Keterangan**

Gambar Kiri (pH by Grade):
- Grade Low memiliki variasi pH yang besar (3-9) dengan median sekitar 6.5.
- Grade Medium dan High memiliki pH yang lebih stabil di kisaran 6-7.

Gambar Kanan (Temperature by Grade):
- Grade Low menunjukkan variasi temperatur yang besar (35-70°C) dengan outlier di 90°C.
- Grade Medium dan High memiliki rentang temperatur lebih konsisten (35-45°C).

3. **Menampilkan grafik semua parameter terhadap grade.**

![image](https://github.com/user-attachments/assets/554d6e47-db0f-4ab7-8c62-9e73681919fc)


**Keterangan**

Taste by Grade
- Taste 1 (enak) banyak ditemukan pada Grade (Low) dan Grade (High).
- Taste 0 (tidak enak) lebih banyak pada Grade (Medium).

Odor by Grade 
- Odor 0 (bau tidak sedap) mendominasi Grade (Medium) dan Grade (Low).
- Odor 1 (normal) mendominasi Grade (High).

Fat by Grade
- Fat 0 (rendah) dominan di Grade (Medium).
- Fat 1 (tinggi) lebih banyak muncul pada Grade (Low) dan Grade (High).

Turbidity by Grade
- Turbidity 0 (rendah) banyak pada Grade (Medium).
- Turbidity 1 (tinggi) banyak pada Grade (Low) dan Grade (High).

Colour by Grade
- Colour 255 mendominasi semua grade, tapi Grade (Low) paling banyak.
- Nilai warna di bawah 255 cenderung meningkat untuk Grade (High).

## Data Preparation
1. Melakukan mapping label kategori ke angka, mengubah nilai kolom `Grade` dari teks ke angka.
2. Menampilkan jumlah data `Grade` seetelah konversi.
![image](https://github.com/user-attachments/assets/072e4871-2d7e-45b2-a519-511e72bac175)

3. Memisahkan fitur (x) dan label (y)
   - fitur (x) mencakup semua kolom kecuali `Grade`, sedangkan label (y) adalah kolom `Grade`.
   - Tujuan: Memisahkan variabel independen dan dependen untuk pemodelan
4. Menentukan kolom kategorikal dan numerik dari data fitur (x), lalu menampilkan hasil identifikasi kolom.
5. Membuat pipeline untuk transformasi data numerik dan kategorikal.
   - Pipeline numerik menggunakan `StandardScaler` untuk menstandarkan data numerik sehingga memiliki mean 0 dan standar deviasi 1.
   - Pipeline kategorikal menggunakan `OneHotEncoder` untuk mengubah data kategorikal menjadi representasi numerik.
6. Menggabungkan pipeline numerik dan kategorikal ke dalam satu perpocessor untuk memastikan transformasi dilakukan secara paralel pada kolom yang sesuai.
7. Membagi dataset ke dalam data latih (80%) dan data uji (20%) untuk memastikan model dilatih dan diuji pada data yang berbeda untuk menghindari overfitting.

![image](https://github.com/user-attachments/assets/bad2ce77-c045-40db-9196-d561a803f4a0)

## Modeling
Pada tahap ini penulis menggunakan dua algoritma machine learning yaitu Random Forest (RF) dan Support Vector Machine (SVM) untuk menyelesasikan permasalahan klasifikasi dalam Milk Quality Dataset yang diimplementasikan menggunakan pipeline untuk memastikan proses preprocessing dan pelatihan model dilakukan secara terintegrasi. 

RF adalah algoritma ensemble yang menggunakan banyak pohon keputusan untuk menghasilkan prediksi yang stabil dan akurat. RF memiliki kelebihan seperti ketahanan terhadap overfitting dan kemampuan menangani data dengan missing values, tetapi membutuhkan lebih banyak sumber daya komputasi.

Pipeline untuk model RF menggunakan  `RandomForestClassifier` dari library `sklearn.ensemble`. Disini menggunakan parameter seed `random_state=42`. Pipeline ini terdiri dari preprocessing data `preprocessor` dan pelatihan model. Data pelatihan (`X_train`, `y_train`) digunakan untuk melatih model dengan `fit`, sedangkan prediksi dilakukan pada data pengujian (`X_test`) menggunakan `predict`.

![image](https://github.com/user-attachments/assets/76642aae-3d1d-4434-bb6c-8272330aaa37)

![image](https://github.com/user-attachments/assets/1fd673ec-4b15-482a-bddc-c62299548743)

Sedangkan, pipeline SVM menggunakan `SVC` dari `sklearn.svm`. Pipeline ini juga terdiri dari preprocessing data dan pelatihan model. Disini menggunakan parameter seed juga yaitu `random_state=42`. Parameter `probability=True` digunakan untuk memungkinkan keluaran probabilitas prediksi. SVM bekerja dengan mencari hyperplane optimal untuk memisahkan kelas dalam data dan sangat efektif untuk data berdimensi tinggi. Namun, SVM sensitif terhadap noise dan membutuhkan tuning parameter seperti `C` dan `gamma`.

![image](https://github.com/user-attachments/assets/4a3102b4-34a2-4a2e-aecc-4c2af6fbb4a9)

![image](https://github.com/user-attachments/assets/4a822704-cd4b-4306-aaa4-5249c6cd2a40)

Setelah semua model dijalankan, penulis memilih algoritma Random Forest sebagai model terbaik yang akan digunakan sebagai solusi untuk memprediksi kualitas susu. Karena, model ini memiliki akurasi dan skor F1 tertinggi dibandingkan SVM, serta kesalahan klasifikasi pada matriks confusion yang lebih kecil dibanding model lainnya. Penjelasan lebih lengkap mengenai alasan ini ada pada bagian **evaluation**.

## Evaluation
Pada tahap evaluasi ini penulis menggunakan beberapa metrik untuk mengukur performa model. yaitu `accuracy`, `precission`, `recall` dan `F1-score`. Disini penulis memakai confusion matrix untuk memberikan visualisasi mengenai hasil klasifikasi model secara lebih rinci.

### **Metrik Evaluasi yang Digunakan**
1. **Accuracy:** Mengukur presentase prediksi yang benar dibandingkan dengan total data.

![image](https://github.com/user-attachments/assets/8a370777-6d95-44e8-8f81-bbd6ecb5d0b6)

2. **Precission:** Mengindikasikan seberapa banyak prediksi positif yang benar dibandingkan dengan total prediksi positif. Precision penting jika kesalahan positif palsu (false positive) perlu diminimalkan.
   
![image](https://github.com/user-attachments/assets/f971383d-6f41-4eed-a920-3999abc4c8e7)

3. **Recall:** Mengukur kemampuan model dalam mendeteksi seluruh sampel positif yang sebenarnya. Recall penting jika kesalahan negatif palsu (false negative) harus diminimalkan.

![image](https://github.com/user-attachments/assets/6b1cbdba-68fc-40d5-8e84-1448f4df8ab4)

4. **F1-score:** Mengombinasikan precision dan recall menjadi satu nilai rata-rata tertimbang, sehingga memberikan gambaran tentang keseimbangan performa model di semua kelas.

![image](https://github.com/user-attachments/assets/101a1847-09ae-40a1-885d-3a5cc0e9d39f)

### **Hasil Evaluasi Model**

![image](https://github.com/user-attachments/assets/2123e0dc-86da-48d7-9b38-ee56cf2b9e7f)

#### **Random Forest**
- **Accuracy**: 99.53% — Model ini menunjukkan akurasi sangat tinggi, hampir semua sampel diklasifikasikan dengan benar.
- **F1-score (weighted)**: 99.53% — Menunjukkan keseimbangan yang sangat baik antara precision dan recall di semua kelas.
- **Precision** untuk kelas "Low (0)" dan "Medium (1)" adalah 100%, sedangkan untuk kelas "High (2)" adalah 98%.
- **Recall** untuk semua kelas mendekati atau mencapai 100%.
- **F1-score** menunjukkan performa konsisten di semua kelas, dengan nilai mendekati atau mencapai 1.00.

![image](https://github.com/user-attachments/assets/a18287a0-13e2-46d9-bbe0-ff33d52c0f78)

- **Confusion Matrix**:
  - **Low (0)**: 85 sampel diklasifikasikan dengan benar, 1 salah ke "High (2)".
  - **Medium (1)**: Semua 75 sampel diklasifikasikan dengan benar tanpa kesalahan.
  - **High (2)**: Semua 51 sampel diklasifikasikan dengan benar tanpa kesalahan.

#### **Support Vector Machine (SVM)**

![image](https://github.com/user-attachments/assets/53132c5b-e0d6-45eb-b737-8fe94e855166)

- **Accuracy**: 92.92% — Performa cukup baik, namun lebih rendah dibandingkan dengan Random Forest.
- **F1-score (weighted)**: 92.92% — Menunjukkan keseimbangan yang baik, namun terdapat kekurangan pada beberapa kelas.
- **Precision** untuk kelas "Low (0)" adalah 98%, sedangkan untuk "Medium (1)" dan "High (2)" masing-masing adalah 91% dan 88%.
- **Recall** menunjukkan hasil yang serupa, dengan kelas "High (2)" memiliki recall terendah sebesar 88%.
- **F1-score** juga lebih rendah pada kelas "Medium (1)" dan "High (2)".

![image](https://github.com/user-attachments/assets/cd0044fc-f5f4-49b1-a310-0c63ab42c68b)

**Confusion Matrix (lihat Gambar 1)**:
- **Low (0)**: 84 sampel benar, 1 salah ke "Medium (1)", dan 1 salah ke "High (2)".
- **Medium (1)**: 68 sampel benar, 2 salah ke "Low (0)", dan 5 salah ke "High (2)".
- **High (2)**: 45 sampel benar, 6 salah ke "Medium (1)".

## Kesimpulan
**1. Bagaimana cara memprediksi kualitas susu berdasarkan parameter fisik dan kimia yang ada dalam dataset?**

Berdasarkan hasil analisis dan pemodelan, kualitas susu dapat diprediksi secara efektif dengan menggunakan data parameter fisik (seperti pH, temperature, taste, odor, fat, turbidity, dan colour) melalui pendekatan machine learning. Dengan proses data preparation dan preprocessing yang tepat, serta pemilihan fitur yang relevan, sistem dapat mengklasifikasikan kualitas susu menjadi tiga kategori (low, medium, high) secara otomatis.

**2. Algoritma machine learning apa yang paling efektif dalam memprediksi kualitas susu?**

Dari dua algoritma yang diuji, yaitu Random Forest dan Support Vector Machine (SVM), hasil evaluasi menunjukkan bahwa Random Forest memberikan performa terbaik dengan akurasi dan F1-score paling tinggi (99,53%), dan tingkat kesalahan klasifikasi yang sangat kecil, khususnya pada confusion matrix. Hal ini selaras dengan tujuan bisnis untuk mengetahui algoritma paling efektif dalam prediksi kualitas susu, sehingga Random Forest direkomendasikan sebagai model utama untuk implementasi sistem prediksi kualitas susu.
 
## Referensi
F. Fauzi and A. R. Abdullah, "Milk Quality Prediction Using Machine Learning," ResearchGate, Nov. 2023. [Online]. Available: https://www.researchgate.net/publication/376064637_Milk_Quality_Prediction_Using_Machine_Learning 
