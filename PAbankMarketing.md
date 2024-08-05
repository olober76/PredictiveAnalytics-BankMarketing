# Bank marketing campaigns dataset | Opening Deposit

## Domain Proyek

ini adalah dataset yang mendeskripsikan hasil dari portugal bank marketing campaign

Ini adalah kumpulan data yang menggambarkan hasil kampanye pemasaran bank di Portugal. Kampanye yang dilakukan sebagian besar berbasis panggilan telepon langsung, menawarkan kepada klien bank untuk menempatkan deposito berjangka. Jika setelah semua upaya pemasaran klien setuju untuk menempatkan deposito, variabel target ditandai dengan 'yes', jika tidak maka 'no'.

Sumber data:

[Link Dataset UCI](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

Ada penurunan pendapatan di bank Portugal dan mereka ingin mengetahui tindakan apa yang harus diambil. Setelah penyelidikan, ditemukan bahwa penyebab utamanya adalah karena klien mereka tidak melakukan deposito sesering sebelumnya. Mengetahui bahwa deposito berjangka memungkinkan bank untuk menahan dana untuk jangka waktu tertentu, sehingga bank dapat menginvestasikan dana tersebut dalam produk keuangan dengan keuntungan lebih tinggi untuk menghasilkan keuntungan. Selain itu, bank juga memiliki peluang lebih besar untuk membujuk klien deposito berjangka untuk membeli produk lain seperti reksa dana atau asuransi guna meningkatkan pendapatan mereka lebih lanjut. Akibatnya, bank Portugal ingin mengidentifikasi klien yang memiliki peluang lebih tinggi untuk berlangganan deposito berjangka dan fokus upaya pemasaran pada klien tersebut.

## Businees Understanding

dengan latar belakang di atas problem statement yang bisa di definisikan adalah

### Problem Statement

- Identifikasi klien yang memiliki peluang lebih tinggi untuk berlangganan depostio berjangaka panjang agar bisa fokus upaya pemasaran pada klien tersebut

### Goals

- agar Bank bisa fokus pada klien yang berdeposit berjangka panjang agar tidak mengalami pendapatan minim

### Solution Stataments

Salah satu metode yang dapat menjawab goals di atas adalah dengan membuat model linear regression degan memprediksi apakah ada deposito yang dilakukan berdasarkan nilai-nilai dari fitur-fitur tersebut. Hasilnya akan berupa 0 atau 1. Jadi, kami dapat memutuskan untuk menggunakan model klasifikasi untuk masalah ini.

## Data Understanding

**Dataset:**

Data yang disediakan memiliki 4118 instance dan 21 fitur. Informasinya mengatakan tidak ada nilai nol. Bagaimanapun, kami akan dengan ketat memeriksa setiap fitur dan memeriksa catatan yang mencurigakan serta memanipulasinya.

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

**CATATAN**
Dilakukan juga visulisasi data untuk melihat data data numerik dari setiap atribut dan persebarannya

## Data Preparation

Cleaning data dilakukan untuk mengencek adanya null value apa tidak, karena jika adanya null value akan mempengaruhi pada pembuatan model.
diperlukan juga feature engineering untuk menghandling outliers, standarisasi , dan encoding untuk beberapa data yang bukan numerik di beberapa atribut

## Modelling

disini menggunakan beberapa model yang akan saya coba jalan kan yakni

1. Logistic Regression
2. Decision Tree
3. KNN
4. SVC
5. Naive Bayes

dari semua model setelah di uji, logistic regression memiliki akurasi kedua lebih tinggi (82%) setelah SVM , ini dikarenakan semua atribut memiliki korelasi yang kuat satu sama lain sehingga saya memilih model dengan beberapa parameter tuning

## Evaluasi

evaluasi yang digunakan berupa matriks yang terdiri atas

1. Confusion Matrix
2. recall
3. f1-score
4. Support
5. ROC Curve

evaluasi ini diperlukan untuk melihat model yang diuji setelah mengalami hyper parameter tuning yang telah meningkat akurasinya menjadi 92%
