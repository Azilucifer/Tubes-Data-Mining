# Catatan Eksplorasi Data Mining — Kelompok 3
# Segmentasi Risiko Belajar Siswa (student-por.csv)

## Fase 1: Pemahaman Dataset
- Dataset: student-por.csv dari UCI Machine Learning Repository
- Format CSV dengan separator titik koma (;), bukan koma
- 649 siswa, 33 variabel awal
- Variabel campuran: numerik (age, absences, dsb) dan kategorikal (school, sex, higher, dsb)
- Tidak ada missing value → dataset cukup bersih

## Fase 2: Penentuan Arah Analisis
- Awalnya mempertimbangkan prediksi nilai (G3), tapi diputuskan untuk melakukan segmentasi perilaku
- Alasan: segmentasi memberikan insight tentang pola siswa, bukan sekadar prediksi angka
- G1, G2, G3 dihapus sejak awal agar tidak terjadi data leakage
- Fokus analisis: risiko belajar, motivasi pendidikan, dan dukungan akademik

## Fase 3: Eksplorasi Variabel
- Cek korelasi antar variabel numerik → tidak ada korelasi yang terlalu tinggi
- Cek distribusi variabel penting:
  - studytime: mayoritas siswa belajar 1-2 jam/minggu
  - failures: 85% siswa tidak pernah gagal
  - absences: ada beberapa siswa dengan absensi sangat tinggi (outlier)
  - higher: mayoritas (>90%) ingin lanjut pendidikan tinggi
  - schoolsup: hanya ~10% mendapat dukungan tambahan sekolah

## Fase 4: Percobaan Kombinasi Fitur
- Kandidat 1: studytime, failures, absences_capped, higher_num, schoolsup_num → Silhouette 0.4928
- Kandidat 2: inti 4 variabel (tanpa schoolsup) → Silhouette lebih rendah
- Kandidat 3: tambah paid → distribusi cluster kurang seimbang
- Kandidat 4: tambah internet → tidak menambah kualitas cluster
- Kandidat 5: pakai Medu, Fedu → cluster didominasi latar belakang keluarga, bukan perilaku belajar
- Keputusan: pilih Kandidat 1 karena metrik terbaik dan interpretasi paling jelas

## Fase 5: Percobaan Jumlah Cluster
- k=2: hanya membedakan yang punya dukungan sekolah vs tidak, terlalu general
- k=3: tiga kelompok jelas — rendah risiko, terdukung, tinggi risiko
- k=4: mulai ada cluster yang overlap karakteristiknya
- k=5: terlalu granular, cluster kecil sulit diinterpretasikan
- Keputusan: k=3 berdasarkan Silhouette tertinggi, DBI terendah, dan interpretasi terbaik

## Fase 6: Penanganan Outlier
- Cek outlier absences: 21 siswa di atas batas IQR (>15)
- Capping ke batas atas 15 → outlier menjadi 0
- studytime dan failures tidak dicapping karena ordinal (bukan outlier asli)
- Pertimbangan: mencapping variabel ordinal = mengubah kategori jawaban, tidak valid

## Fase 7: Temuan Tidak Terduga
- Logistic Regression accuracy 100% → awalnya curiga overfitting
  - Setelah dicek: gap train-test = 0, bukan overfitting
  - Alasan: target = label K-Means yang batasnya linear, LR memang model linear
- R² regression sangat rendah (0.058)
  - Ketidakhadiran sulit diprediksi hanya dari 4 fitur
  - Banyak faktor di luar dataset (sakit, jarak rumah, keluarga)
  - Ini bukan kegagalan, tapi keterbatasan yang harus dilaporkan
- Cluster 1 (schoolsup) hanya 66 siswa (10.2%)
  - Bukan masalah → memang sedikit siswa yang dapat dukungan tambahan
  - Justru menunjukkan K-Means bisa menangkap pola minoritas

## Fase 8: Interpretasi Cluster
- Cluster 0 (507 siswa): Risiko rendah — failures rendah, absences rendah, higher=1, schoolsup=0
- Cluster 1 (66 siswa): Terdukung — schoolsup=1 jadi pembeda utama
- Cluster 2 (76 siswa): Risiko tinggi — failures tinggi, absences tinggi, higher=0
- Penamaan cluster berdasarkan profil rata-rata, bukan label subjektif

## Fase 9: Pengembangan Dashboard
- Pilih Streamlit karena Python-native dan cepat untuk prototype
- 6 halaman: ringkasan, prediksi, profil, visualisasi, evaluasi, rekomendasi
- Semua data diambil dari model bundle (pkl), tidak ada angka hardcoded
- Input absences dibatasi 0-15 sesuai batas capping IQR
