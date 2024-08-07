# Bank marketing campaigns dataset | Opening Deposit

## Domain Proyek

ini adalah dataset yang mendeskripsikan hasil dari portugal bank marketing campaign

Ini adalah kumpulan data yang menggambarkan hasil kampanye pemasaran bank di Portugal. Kampanye yang dilakukan sebagian besar berbasis panggilan telepon langsung, menawarkan kepada klien bank untuk menempatkan deposito berjangka. Jika setelah semua upaya pemasaran klien setuju untuk menempatkan deposito, variabel target ditandai dengan 'yes', jika tidak maka 'no'.

Ada penurunan pendapatan di bank Portugal dan mereka ingin mengetahui tindakan apa yang harus diambil. Setelah penyelidikan, ditemukan bahwa penyebab utamanya adalah karena klien mereka tidak melakukan deposito sesering sebelumnya. Mengetahui bahwa deposito berjangka memungkinkan bank untuk menahan dana untuk jangka waktu tertentu, sehingga bank dapat menginvestasikan dana tersebut dalam produk keuangan dengan keuntungan lebih tinggi untuk menghasilkan keuntungan. Selain itu, bank juga memiliki peluang lebih besar untuk membujuk klien deposito berjangka untuk membeli produk lain seperti reksa dana atau asuransi guna meningkatkan pendapatan mereka lebih lanjut. Akibatnya, bank Portugal ingin mengidentifikasi klien yang memiliki peluang lebih tinggi untuk berlangganan deposito berjangka dan fokus upaya pemasaran pada klien tersebut.

**WHY IS IT IMPORTANT USING MACHINE LEARNING**

masalah ini bisa diselesaikan machine learning karena dengan machine learning model bisa untuk menyortir beberapa atribut dengan tipe numerical data (dalam kasus ini yaitu nasabah yang melakukan deposit jangka panjang) yang di inginkan sehingga perlu adanya klasifikasi pada permasalahan ini

## Businees Understanding

dengan latar belakang di atas problem statement yang bisa di definisikan adalah

### Problem Statement

- Bagaimana cara mengidentifikasi klien yang memiliki peluang lebih tinggi untuk berlangganan depostio berjangaka panjang?

### Goals

- Dengan mengetahui adanya klien yang berdeposit jangka panjang bank dapat menginvestasikan dalam produk keuangan dengan keuntungan lebih tinggi untuk mendapatkan keuntungan. Selain itu, bank juga memiliki peluang yang lebih baik untuk meyakinkan klien deposito berjangka untuk membeli produk lain seperti dana atau asuransi guna meningkatkan pendapatan mereka lebih lanjut.

### Solution Stataments

algoritma klasifikasi yang secara otomatis dapat mengklasifikasikan prospek Bank mengenai kemungkinan membuka deposito berjangka di bank mereka. saya akan membuat algoritma logistic regression dan juga memberikan wawasan yang saya peroleh dari dataset tersebut. Selain itu, saya akan membantu mereka mempersempit prospek ke dalam saluran pemasaran dan pada akhirnya membuka deposito berjangka. parameter keberhasilan algoritma logistic regression akan diukur oleh metrix evaluasi seperti confussion matrix, f1 score, akurasi, dan precision

## Data Understanding

Sumber data:

[Link Dataset UCI](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

**Dataset:**

Data yang disediakan memiliki 4118 instance dan 21 fitur. Informasinya mengatakan tidak ada nilai nol. Bagaimanapun, perlu dilakukan pengawasan ketat di setiap fitur dan memeriksa catatan yang mencurigakan serta memanipulasinya.

**Atribut:**
**Data klien bank:**

1. **Age**: Usia prospek (numerik)
2. **Job**: Jenis pekerjaan (Kategorikal)
3. **Marital**: Status pernikahan (Kategorikal)
4. **Education**: Kualifikasi pendidikan prospek (Kategorikal)
5. **Default**: Apakah prospek memiliki kredit yang bermasalah (tidak dibayar) (Kategorikal)
6. **Housing**: Apakah prospek memiliki pinjaman perumahan? (Kategorikal)
7. **Loan**: Apakah prospek memiliki pinjaman pribadi? (Kategorikal)

**Terkait dengan kontak terakhir dari kampanye saat ini:**

8. **Contact**: Jenis komunikasi kontak (Kategorikal)
9. **Month**: Bulan kontak terakhir tahun (Kategorikal)
10. **Day_of_week**: Hari kontak terakhir dalam minggu (Kategorikal)
11. **Duration**: Durasi kontak terakhir, dalam detik (numerik).

**Catatan penting:** Durasi sangat memengaruhi target output (misalnya, jika durasi=0 maka y='no'). Namun, durasi tidak diketahui sebelum panggilan dilakukan. Juga, setelah panggilan berakhir, y jelas diketahui. Oleh karena itu, input ini hanya harus disertakan untuk tujuan pembandingan dan harus diabaikan jika tujuannya adalah memiliki model prediktif yang realistis.

**Atribut lainnya:**

12. **Campaign**: Jumlah kontak yang dilakukan selama kampanye ini dan untuk klien ini (numerik)
13. **Pdays**: Jumlah hari yang berlalu setelah klien terakhir kali dihubungi dari kampanye sebelumnya (numerik; 999 berarti klien belum pernah dihubungi sebelumnya)
14. **Previous**: Jumlah kontak yang dilakukan sebelum kampanye ini dan untuk klien ini (numerik)
15. **Poutcome**: Hasil dari kampanye pemasaran sebelumnya (kategorikal)

**Atribut konteks sosial dan ekonomi:**

16. **Emp.var.rate**: Tingkat variasi pekerjaan - indikator kuartalan (numerik)
17. **Cons.price.idx**: Indeks harga konsumen - indikator bulanan (numerik)
18. **Cons.conf.idx**: Indeks kepercayaan konsumen - indikator bulanan (numerik)
19. **Euribor3m**: Suku bunga Euribor 3 bulan - indikator harian (numerik)
20. **Nr.employed**: Jumlah karyawan - indikator kuartalan (numerik)

**Variabel output (target yang diinginkan):**

21. **Y** - Apakah klien berlangganan deposito berjangka? (biner: 'yes', 'no')

**VISUALISASI**
Dilakukan juga visulisasi data untuk melihat data data numerik dari setiap atribut dan persebarannya yang dimana akan menjadi bahan pertimbangan dalam melakukan feature engineering

1. Duration of calls vs Job roles

![Duration of calls vs Job roles](https://github.com/user-attachments/assets/a10ea559-7cb3-4287-a41e-b6f84e223100)

- Prospek yang tidak melakukan deposito memiliki durasi panggilan yang lebih pendek.
- Jika dibandingkan rata-ratanya, pekerja kasar dan pengusaha memiliki durasi panggilan yang tinggi, sedangkan pelajar dan pensiunan memiliki durasi panggilan rata-rata yang rendah.
- Sebagian besar prospek berasal dari klien wiraswasta dan orang-orang manajemen.

2. Campaign vs Duration Calls

![Campaign vs Duration Calls](https://github.com/user-attachments/assets/9eae2865-390a-4820-a957-e6ea2c689f6d)

- Semakin lama durasi panggilan, semakin tinggi probabilitas untuk melakukan deposito.
- Durasi panggilan menurun seiring dengan berjalannya waktu kampanye.
- Terdapat banyak prospek positif pada hari-hari awal kampanye.

3. Campaign vs Month

![Campaign vs Month](https://github.com/user-attachments/assets/a8c6edef-71db-407c-834f-76b811cd1552)

- Kampanye sebagian besar terkonsentrasi pada awal periode bank (Mei, Juni, dan Juli).
- Biasanya periode pendidikan dimulai pada waktu tersebut sehingga ada kemungkinan orang tua membuat deposito atas nama anak-anak mereka.
- Mereka juga melakukan kampanye di akhir periode bank.

4. Distribution of Quarterly Indicators

![Distribution of Quarterly Indicators](https://github.com/user-attachments/assets/a9e8c937-18e8-4eaa-bcc1-b2394c97bf97)

- Terlihat adanya variasi karyawan yang tinggi yang menandakan bahwa kampanye dilakukan saat terjadi pergeseran pekerjaan yang tinggi akibat kondisi ekonomi.
- Indeks harga konsumen yang baik menunjukkan bahwa prospek memiliki daya beli yang baik untuk barang dan jasa, yang mungkin menjadi alasan untuk mendorong mereka melakukan deposito dan menanamkan ide menabung.
- Indeks kepercayaan konsumen cukup rendah karena mereka tidak memiliki banyak kepercayaan pada ekonomi yang berfluktuasi.
- Suku bunga Euribor 3 bulan adalah suku bunga di mana sejumlah bank Eropa meminjamkan dana dalam bentuk euro satu sama lain dengan jangka waktu 3 bulan. Dalam kasus kami, suku bunga pinjaman tinggi.
- Jumlah karyawan juga mencapai puncaknya, yang dapat meningkatkan indeks pendapatan mereka. Hal ini mungkin menjadi alasan kampanye menargetkan prospek yang bekerja untuk melakukan deposito.

5. Marital Status vs Price index

![Marital Status vs Price index](https://github.com/user-attachments/assets/ad5240cf-d0cd-471a-800a-be6f4f8b14f8)

- Terdapat perbedaan yang sangat kecil di antara indeks harga.
- Prospek yang sudah menikah memiliki keunggulan yang signifikan karena mereka memiliki indeks yang berkontribusi sebagai pasangan.

6. Positive deposits vs attributes

![Positive deposits vs attributes](https://github.com/user-attachments/assets/c09fc062-821f-4a1b-885a-ee96799647e6)

7. Correlation plot of attributes

![Correlation plot of attributes](https://github.com/user-attachments/assets/ff1cf3e3-7774-4235-ace8-a94c41e25af3)

- Prospek yang sudah menikah membuat deposito tinggi diikuti oleh yang lajang.
- Banyak deposito dilakukan pada bulan Mei karena itu adalah awal periode bank.
- Prospek yang bekerja di posisi administrasi membuat deposito, diikuti oleh teknisi dan pekerja kasar.
- Prospek yang memiliki setidaknya gelar universitas membuat deposito diikuti oleh lulusan sekolah menengah.

Proses data Understanding dilakukan agar tidak mempengaruhi proses klasifikasi nantinya, beberapa proses pengecekan data yang saya lakukan yaitu

**Null Values Checking**

dilakukan checking null values untuk melihat adanya indikasi data bernilai 0 di beberapa atribut

**Duplicating Data**
dilakukan untuk melihat duplicating data di setiap instances

## Data Preparation

Bebebarapa tahapan yang diperlukan sebagai fitur fitur yang dipilih untuk pembentukan model klasifikasi yaitu

**Handling Outliers**
proses inii diperlakukan agar nanti model tidak terpengaruh nilai nilai ekstrim

diperlukan juga feature engineering dan standarisasi , dan encoding untuk beberapa data yang bukan numerik di beberapa atribut

1. **Education - category clubbing** : untuk mengelompokan data yang memiliki kategori dari basic9y sampai dengan middle school
2. **Encoding - Month and Day of Week** : mengenkodekan kategori Month and Day of Week ke dalam respective numbers
3. **Encoding 999 in pdays as 0** : mengenkodekan nilai 999 dalam fitur pdays menjadi 0 (klien yang belum dikontak dalam campaign sebelumnya)
4. **Ordinal Number Encoding** : Mengumbah fitur yang mempunyai nilai string seperti 'yes', 'no', dan 'unknown' menjadi yes:1,no:0 and unknown:-1
5. **Ordinal Encoding** : menghapus fitur yang memiliki masukan string dan digantikan dengan ordinal number encoding (one hot encoding)
6. **Frequency Encoding** : menggunakan frekuensi ecoding pada fitur 'job' dan 'education' menjadi key value pairs berdasarkan frekuensinya
7. **Target Guided Ordinal Encoding** : mendefinisikan target Y , yaitu marital feature yang di enkodekan (diubah masukan datanya menjadi key:value pairs seperti {'divorced': 0, 'married': 1, 'single': 2, 'unknown': 3})
8. **Standardization of numerical Variables** : melakukan standarisasi pada data yang sudah enkodekan , disini saya menggunakan fungsi StandardScaler(),yaitu proses mengubah skala data sehingga memiliki rata-rata 0 dan deviasi standar 1.
9. **Feature Selection** : melihat mana fitur yang penting yangn nantinya akan di pangkas fiturnya agar model berjalan dengan baik
10. **Train and Test Split (80:20)** : memisah data untuk dimasukan kedalam train dan test dengan rasio yang 80 banding 20

## Modelling

disini menggunakan beberapa model yang akan saya coba jalan kan yakni

1. Logistic Regression Accuracy: 0.882482599781859
2. Decision Tree Accuracy: 0.6428312903816458
3. KNN Accuracy: 0.8746929484882704
4. SVC Accuracy: 0.9188718011316903
5. Naive Bayes Accuracy: 0.8191811374646486

Berikut adalah penjelasan tahapan kerja untuk setiap algoritma yang digunakan dalam kode tersebut:

1. **Logistic Regression (Regresi Logistik)**:

   - **Inisialisasi**: Objek `LogisticRegression` diinisialisasi dengan parameter `random_state=0` untuk memastikan hasil yang konsisten di setiap run.
   - **Pelatihan**: Model regresi logistik dilatih pada data pelatihan `X` (fitur) dan `y` (label).
   - **Prediksi**: Model memprediksi probabilitas kejadian suatu kelas berdasarkan fitur input.
   - **Evaluasi**: Akurasi diuji menggunakan cross-validation (10-fold cross-validation) dan hasil rata-rata akurasi dilaporkan.

2. **Decision Tree (Pohon Keputusan)**:

   - **Inisialisasi**: Objek `DecisionTreeClassifier` diinisialisasi.
   - **Pelatihan**: Model pohon keputusan dilatih pada data `X` dan `y`.
   - **Prediksi**: Model membagi data berdasarkan fitur yang memberikan informasi paling banyak tentang kelas target dan memprediksi kelas berdasarkan aturan yang diturunkan.
   - **Evaluasi**: Akurasi diuji menggunakan cross-validation (10-fold cross-validation) dan hasil rata-rata akurasi dilaporkan.

3. **K-Nearest Neighbors (KNN)**:

   - **Inisialisasi**: Objek `KNeighborsClassifier` diinisialisasi.
   - **Pelatihan**: Model KNN tidak melakukan pelatihan eksplisit, namun menyimpan data pelatihan untuk digunakan saat prediksi.
   - **Prediksi**: Model memprediksi kelas untuk sampel baru dengan melihat kelas dari k-tetangga terdekat dalam data pelatihan.
   - **Evaluasi**: Akurasi diuji menggunakan cross-validation (10-fold cross-validation) dan hasil rata-rata akurasi dilaporkan.

4. **Support Vector Classifier (SVC)**:

   - **Inisialisasi**: Objek `SVC` diinisialisasi.
   - **Pelatihan**: Model SVC dilatih untuk menemukan hyperplane yang memisahkan kelas-kelas dalam fitur `X` dengan margin terbesar.
   - **Prediksi**: Model memprediksi kelas berdasarkan sisi mana dari hyperplane sampel baru berada.
   - **Evaluasi**: Akurasi diuji menggunakan cross-validation (10-fold cross-validation) dan hasil rata-rata akurasi dilaporkan.

5. **Naive Bayes (BernoulliNB)**:
   - **Inisialisasi**: Objek `BernoulliNB` diinisialisasi.
   - **Pelatihan**: Model Naive Bayes dilatih dengan menghitung probabilitas fitur biner terhadap setiap kelas.
   - **Prediksi**: Model memprediksi kelas dengan menghitung probabilitas posterior berdasarkan teorema Bayes.
   - **Evaluasi**: Akurasi diuji menggunakan cross-validation (10-fold cross-validation) dan hasil rata-rata akurasi dilaporkan.

**Proses Evaluasi dengan Cross-Validation**:

- **Cross-Validation (10-Fold)**: Dataset dibagi menjadi 10 bagian (folds). Model dilatih pada 9 bagian dan diuji pada 1 bagian, dan proses ini diulang 10 kali sehingga setiap bagian menjadi set uji satu kali.
- **Scoring**: Setiap model dievaluasi berdasarkan akurasi prediksi pada set uji.
- **Rata-rata Akurasi**: Rata-rata akurasi dari 10 fold dilaporkan untuk setiap model, memberikan gambaran umum tentang kinerja model pada data tersebut.

Tujuan utama dari pendekatan ini adalah untuk membandingkan kinerja berbagai algoritma klasifikasi pada dataset yang sama dan memilih algoritma yang memberikan akurasi terbaik untuk prediksi apakah prospek akan membuka deposito berjangka.

dari semua model setelah di uji, logistic regression memiliki akurasi kedua lebih tinggi (88%) setelah SVM , ini dikarenakan logistic regression lebih cocok dalam mengklasifikasikan dua jenis label saja sehingga saya memilih model ini dengan beberapa parameter tuning

tahapan Logisctic classifier yang dilakukan yaitu

Dengan melakukan pencarian hyperparameter (hyperparameter tuning) untuk model regresi logistik menggunakan `GridSearchCV` dari pustaka `scikit-learn`.

- parameter grid (`param_grid`) didefinisikan dengan nilai `C` (regularization strength) yang diambil dari rentang logaritmik antara 10^-4 hingga 10^4 dalam 50 nilai, serta penalti (`penalty`) yang bisa berupa 'l1' atau 'l2'.

- menemukan kombinasi hyperparameter terbaik melalui cross-validation sebanyak 5 kali (`cv=5`).

- Model terbaik dari pencarian ini kemudian dilatih dengan data pelatihan (`X_train` dan `y_train`) yang sudah di definisikan setelah melalui proses feature engineering .

- Hasil dari model terbaik dicetak , serta akurasi rata-rata model pada data uji (`X_test` dan `y_test`).

- Setelah itu, sebuah model regresi logistik baru dibuat dengan nilai tertentu yang ditemukan dari akan dilatih kembali dengan data pelatihan. Akurasi model ini pada set data uji dicetak untuk menilai performanya.

## Evaluasi

evaluasi ini diperlukan untuk melihat model yang diuji setelah mengalami hyper parameter tuning yang telah meningkat akurasinya menjadi 92%

evaluasi yang digunakan berupa matriks yang terdiri atas

![Evaluation](https://github.com/user-attachments/assets/6ec290fe-e08f-4266-9984-441b7f85d490)

1. **Confusion Matrix**

True Positives (TP): 199 (model memprediksi positif dengan benar)
True Negatives (TN): 6390 (model memprediksi negatif dengan benar)
False Positives (FP): 148 (model memprediksi positif tetapi salah)
False Negatives (FN): 376 (model memprediksi negatif tetapi salah)

2. recall

0.98 (98% dari total data negatif terprediksi dengan benar) pada class 0 (Negative Class) dan Recall: 0.35 (35% dari total data positif terprediksi dengan benar) pada Class 1 (Positive Class)

3. f1-score

0.96 (harmonik rata-rata dari precision dan recall) pada class 0 (Negative Class) dan 0.43 (harmonik rata-rata dari precision dan recall) pada Class 1 (Positive Class)

4. Macro Average:

- Precision: 0.76 (rata-rata precision untuk kedua kelas)
- Recall: 0.66 (rata-rata recall untuk kedua kelas)
- F1-score: 0.70 (rata-rata f1-score untuk kedua kelas)

5. Weigthed Average

- Precision: 0.91 (rata-rata precision yang mempertimbangkan jumlah instance di setiap kelas)
- Recall: 0.93 (rata-rata recall yang mempertimbangkan jumlah instance di setiap kelas)
- F1-score: 0.92 (rata-rata f1-score yang mempertimbangkan jumlah instance di setiap kelas)

4. ROC Curve

![Gambar ROC](https://github.com/user-attachments/assets/f4351dbc-6843-4fea-bc67-ed5f40ae0f14)

Dari kurva ROC,kita dapat menyimpulkan bahwa model logisctic regression telah mengklasifikasikan prospek yang melakukan deposit dengan benar, bukan memprediksi positif palsu. Semakin kurva ROC (merah) terletak di sisi kiri atas, semakin baik model kita. Kita dapat memilih nilai apa pun antara 0,8 hingga 0,9 untuk nilai ambang batas yang dapat menghasilkan hasil positif yang sebenarnya.

**KESIMPULAN**

Dari bagian EDA dan pemilihan model logisctic regression , kita dapat mengidentifikasi dengan jelas bahwa durasi memainkan peran penting dalam menentukan hasil dari dataset kita. Jelas bahwa semakin banyak lead yang tertarik untuk memulai deposit, jumlah panggilan akan lebih tinggi dan durasi panggilan akan lebih lama daripada rata-rata. Kami juga menemukan bahwa pekerjaan dan pendidikan juga berperan penting dan sangat mempengaruhi hasil.

Berikut adalah beberapa rekomendasi untuk bank yang dapat membantu meningkatkan tingkat deposit:

- Klasifikasikan peran pekerjaan berdasarkan tingkatan korporat dan dekati semua karyawan tingkatan 1 dalam beberapa hari setelah kampanye dimulai.
- Dengarkan lead dan kumpulkan lebih banyak informasi untuk memberikan rencana deposit terbaik, yang dapat meningkatkan durasi panggilan dan mengarah pada deposit.
- Mendekati lead pada awal periode baru bank (Mei-Juli) akan menjadi pilihan yang baik karena banyak yang menunjukkan hasil positif dari data sejarah.
- Sesuaikan kampanye sesuai dengan kondisi ekonomi nasional, jangan menyalurkan pengeluaran pada kampanye ketika ekonomi nasional sedang berkinerja buruk.
