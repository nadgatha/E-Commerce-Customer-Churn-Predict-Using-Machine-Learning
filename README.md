# **E-COMMERCE CUSTOMER CHURN PREDICT USING MACHINE LEARNING**

## Latar Belakang
Di era bisnis modern, E-commerce menjadi pilar utama dalam ekonomi global dengan perkembangan teknologi yang pesat. Sebagai sarana transaksi online, E-commerce membuka peluang pertumbuhan yang besar. Dalam konteks ini, retensi pelanggan menjadi krusial untuk kesuksesan E-commerce di tengah persaingan yang ketat.

Faktor-faktor yang menyebabkan pelanggan beralih dalam E-commerce sangat beragam, melibatkan kualitas layanan, pengalaman pengguna, harga, kepuasan produk, komunikasi, perubahan preferensi pelanggan, persaingan pasar, dan ketidakpuasan terhadap kebijakan bisnis. Menjaga pelanggan setia menjadi penting, karena penggantian pelanggan dengan yang baru mengakibatkan biaya yang signifikan.

## Pernyataan Masalah
Perusahaan E-commerce menghadapi tantangan dalam memprediksi dan mencegah pelanggan beralih, sehingga perlu mengembangkan strategi cerdas untuk mengidentifikasi pelanggan yang berisiko. Hal ini memungkinkan upaya promosi yang tepat waktu dan terarah, menghindari kerugian karena kehilangan pelanggan setia, dan mengoptimalkan efektivitas sumber daya promosi.

## Tujuan
Tujuannya adalah mengembangkan kemampuan untuk memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan, memungkinkan fokus upaya retensi pada pelanggan yang terindikasi potensial beralih, dan mengurangi biaya yang tidak perlu.

## Pendekatan Analitis
Pendekatan analitis melibatkan pembuatan, evaluasi, dan implementasi model klasifikasi machine learning yang dapat memprediksi churn pelanggan berdasarkan data historis.

## Metrik Evaluasi
Metrik utama adalah **F2 score**, yang seimbang antara presisi dan recall, dengan fokus mengurangi False Negatives. Metrik ini penting untuk menilai kinerja model, terutama ketika biaya melewatkan pelanggan potensial yang beralih (False Negative) lebih kritis daripada mengidentifikasi secara salah pelanggan yang tidak beralih (False Positive).

## Data Info
Dataset ini mencakup informasi pelanggan dari sebuah perusahaan E-commerce dan bertujuan untuk memahami dan memprediksi perilaku churn pelanggan. Churn merujuk pada pelanggan yang berhenti menggunakan layanan atau produk dari perusahaan.

Dataset ini mencakup berbagai atribut atau fitur yang memberikan gambaran tentang perilaku dan karakteristik pelanggan. Atribut tersebut melibatkan informasi seperti durasi berlangganan (Tenure), perangkat login yang disukai, tier kota, jarak pengiriman, mode pembayaran yang disukai, gender, hingga seberapa sering pelanggan menggunakan aplikasi.

### Kolom Data

| Kolom                        | Jumlah Non-Null | Tipe Data | Deskripsi                                                    |
| ---------------------------- | --------------- | --------- | ------------------------------------------------------------ |
| CustomerID                   | 5630            | int64     | Identifikasi unik untuk setiap pelanggan.                    |
| Churn                        | 5630            | int64     | Indikator biner (0 atau 1) yang menunjukkan status churn pelanggan. 1 menunjukkan churn, 0 menunjukkan retensi. |
| Tenure                       | 5366            | float64   | Jumlah bulan pelanggan telah menggunakan platform E-commerce. |
| PreferredLoginDevice         | 5630            | object    | Perangkat yang lebih disukai oleh pelanggan untuk login (misalnya, Mobile, Desktop). |
| CityTier                     | 5630            | int64     | Klasifikasi tier kota yang menunjukkan tingkat urbanisasi (misalnya, 1, 2, 3). |
| WarehouseToHome              | 5379            | float64   | Jarak dari gudang ke rumah pelanggan (dalam kilometer).      |
| PreferredPaymentMode         | 5630            | object    | Mode pembayaran yang lebih disukai oleh pelanggan (misalnya, Kartu Kredit, E-wallet). |
| Gender                       | 5630            | object    | Jenis kelamin pelanggan (misalnya, Pria, Wanita).            |
| HourSpendOnApp               | 5375            | float64   | Jumlah jam yang dihabiskan pelanggan di aplikasi E-commerce. |
| NumberOfDeviceRegistered     | 5630            | int64     | Jumlah perangkat yang didaftarkan oleh pelanggan.           |
| PreferedOrderCat             | 5630            | object    | Kategori pesanan yang lebih disukai oleh pelanggan (misalnya, Fashion, Elektronik). |
| SatisfactionScore            | 5630            | int64     | Skor kepuasan yang diberikan oleh pelanggan.                |
| MaritalStatus                | 5630            | object    | Status pernikahan pelanggan (misalnya, Single, Menikah).    |
| NumberOfAddress              | 5630            | int64     | Jumlah alamat yang tercatat untuk pelanggan.                |
| Complain                     | 5630            | int64     | Indikator biner (0 atau 1) yang menunjukkan apakah pelanggan pernah mengajukan keluhan. 1 menunjukkan keluhan. |
| OrderAmountHikeFromlastYear  | 5365            | float64   | Persentase kenaikan jumlah pesanan dibandingkan tahun sebelumnya. |
| CouponUsed                   | 5374            | float64   | Jumlah kupon yang digunakan oleh pelanggan.                  |
| OrderCount                   | 5372            | float64   | Total jumlah pesanan yang dilakukan oleh pelanggan.          |
| DaySinceLastOrder            | 5323            | float64   | Jumlah hari sejak pesanan terakhir pelanggan.               |
| CashbackAmount               | 5630            | float64   | Jumlah cashback yang diterima oleh pelanggan.               |

## Data Cleaning

1. **Iterative Imputer:**
   - Terdapat beberapa kolom dengan nilai yang kosong (null), seperti `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`, dan `DaySinceLastOrder`.
   - Penggunaan *iterative imputer* dapat menjadi solusi untuk mengisi nilai-nilai yang hilang dengan estimasi yang lebih akurat berdasarkan pola data keseluruhan.

2. **Analisis Outliers:**
   - Melakukan analisis outliers pada kolom-kolom numerik seperti `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`, dan `DaySinceLastOrder`.
   - Identifikasi dan penanganan outliers penting untuk mencegah pengaruh yang berlebihan pada hasil analisis dan model prediktif.

3. **Deteksi dan Penanganan Duplikat:**
   - Memeriksa keberadaan duplikat dalam dataset untuk mencegah bias yang tidak diinginkan pada analisis.
   - Jika terdapat duplikat, langkah-langkah penanganan duplikat akan diambil untuk memastikan data yang bersih dan akurat.

4. **Pengelompokan Data:**
    - Pengelompokan ini bertujuan untuk mempermudah analisis, mengeksplorasi pola-pola khusus dalam kelompok tertentu, dan menyederhanakan pemahaman terhadap data.

**Langkah-Langkah Pengolahan Data:**
- Iterative imputer akan digunakan untuk mengisi nilai yang hilang pada kolom numerik.
- Analisis outliers akan melibatkan identifikasi nilai-nilai yang ekstrem dan mengambil tindakan yang sesuai.
- Deteksi duplikat akan dilakukan dan data duplikat akan dihapus atau diatasi.
- Pengelompokan data akan memanfaatkan kategori-kategori yang relevan untuk memfasilitasi analisis lebih lanjut.

Data cleaning ini menjadi tahap penting dalam mempersiapkan dataset sebelum dilibatkan dalam analisis dan pembangunan model prediktif.


## Analisis Eksploratif Data (EDA) Faktor-faktor Churn Pelanggan
Dalam menjalankan Analisis Eksploratif Data (EDA) pada dataset pelanggan E-commerce, fokus utama adalah untuk memahami faktor-faktor yang dapat mempengaruhi keputusan pelanggan dalam melakukan churn. Adapun langkah-langkah umum yang dilakukan mencakup evaluasi durasi berlangganan, jenis perangkat login, jarak pengiriman, preferensi mode pembayaran, durasi penggunaan aplikasi, kategori pesanan yang lebih disukai, skor kepuasan, dan status pernikahan pelanggan. Analisis ini bertujuan untuk menyajikan gambaran yang komprehensif dan memfasilitasi pengambilan keputusan strategis untuk mengurangi tingkat churn dalam platform E-commerce tersebut.

## Pemodelan dan Seleksi Model

Dalam tahap pemodelan, digunakan pendekatan *pipeline* yang melibatkan *Iterative Imputer* untuk mengisi nilai-nilai yang hilang, *One-Hot Encoding (OHE)* untuk variabel kategorikal, dan *Standard Scaler* untuk menormalkan skala data. Setelahnya, dilakukan pemilihan model dan evaluasi menggunakan metrik F2 score.

1. **Pipeline Iterative Imputer, OHE, dan Scaler:**
   - *Iterative Imputer* mengisi nilai-nilai yang hilang dengan estimasi yang lebih akurat.
   - *One-Hot Encoding (OHE)* mengonversi variabel kategorikal menjadi format numerik.
   - *Robust Scaler* menormalkan skala data.

2. **Seleksi Model dan Benchmarking:**
   - Beberapa model machine learning, seperti *Logistic Regression, Random Forest, dan XGBoost*, dievaluasi untuk memilih model terbaik.
   - Evaluasi model menggunakan metrik *F2 score*, dengan penekanan lebih pada recall untuk menanggulangi *False Negatives*.

3. **Optimasi Model dengan Resampling (ROS dan RUS):**
   - Dilakukan resampling menggunakan teknik *Random Oversampling (ROS)* dan *Random Undersampling (RUS)* untuk menangani ketidakseimbangan kelas.
   - ROS menambah sampel dari kelas minoritas, sedangkan RUS mengurangi sampel dari kelas mayoritas.

4. **Pemilihan Model Terbaik:**
   - Model terbaik akan dipilih berdasarkan nilai F2 score tertinggi setelah dilakukan benchmarking dengan ROS dan RUS.
   - Pemilihan variabel atau *features* yang signifikan juga dieksplorasi untuk meningkatkan kinerja model.

5. **Final Model:**
   - Model terbaik yang dihasilkan dari proses ini yang menggunakan *XGBoost dengan teknik ROS* akan dijadikan sebagai model final.

Proses ini bertujuan untuk memberikan solusi prediktif yang optimal dalam mengidentifikasi pelanggan yang berpotensi melakukan churn di platform E-commerce, dengan penanganan ketidakseimbangan kelas dan optimasi kinerja model melalui resampling.

## Kesimpulan

### Evaluasi Dampak Penggunaan Model Machine Learning

Berdasarkan Confusion Matrix pada hasil prediksi model final, diperoleh hasil sebagai berikut:

- True Positive (Pelanggan yang actual churn dan diprediksi churn): 172 orang
- False Negative (Pelanggan yang actual churn tetapi diprediksi tidak churn): 18 orang
- False Positive (Pelanggan yang actual tidak churn tetapi diprediksi churn): 12 orang
- True Negative (Pelanggan yang actual tidak churn dan diprediksi tidak churn): 924 orang

### Dampak Keuangan

- Biaya retensi per pelanggan = $27,4
- Biaya akibat kehilangan pelanggan (Churn) = $274 per bulan per pelanggan

#### Estimasi Biaya Tanpa Model Machine Learning
- Total biaya tanpa model machine learning: $82912,4 per bulan
- Calon pelanggan yang kita berikan program retensi: 1126 orang (total pelanggan)
- Calon pelanggan yang aktualnya churn: 190 orang

#### Estimasi Biaya dengan Model Machine Learning
- Total biaya dengan model machine learning: $57293,46 per bulan
- Calon pelanggan yang akan diberikan program retensi: 184 orang (True Positive + False Positive)
- Calon pelanggan yang aktualnya churn: 190 orang

### Manfaat Penggunaan Model Machine Learning
- Total penurunan potensial loss: $82912.4 - $57101.6 = $25619 per bulan
- Persentase penurunan: 30,90%

**Kesimpulan:**
Berdasarkan data uji, model final dapat membantu perusahaan menurunkan kerugian hingga 30,90% dalam sebulan menggunakan machine learning. Penggunaan model ini memungkinkan perusahaan untuk lebih efisien dalam memberikan program retensi kepada pelanggan yang berpotensi churn, sehingga mengurangi biaya yang dikeluarkan tanpa mengurangi efektivitas program tersebut.
