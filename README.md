# Tubes Data Mining — Segmentasi Risiko Belajar Siswa

## Deskripsi Proyek

Proyek Data Mining untuk **segmentasi risiko belajar dan dukungan akademik siswa** menggunakan dataset `student-por.csv`. Model utama adalah **K-Means Clustering** (k=3), dilengkapi Logistic Regression, Gaussian Naive Bayes, dan Linear Regression sebagai analisis pendukung.

## Fitur yang Digunakan

| Variabel | Makna |
|----------|-------|
| `studytime` | Waktu belajar mingguan siswa |
| `failures` | Riwayat kegagalan akademik sebelumnya |
| `absences_capped` | Jumlah ketidakhadiran setelah capping outlier |
| `higher_num` | Motivasi melanjutkan pendidikan tinggi |
| `schoolsup_num` | Dukungan tambahan dari sekolah |

> **Catatan:** Variabel G1, G2, dan G3 (nilai akademik) **tidak digunakan** dalam seluruh proses analisis.

## Hasil Clustering

| Cluster | Nama | Jumlah Siswa |
|---------|------|-------------|
| 0 | Siswa Risiko Belajar Rendah | 507 |
| 1 | Siswa dengan Dukungan Akademik Sekolah | 66 |
| 2 | Siswa Risiko Belajar Tinggi | 76 |

- **Silhouette Score:** 0.4928
- **Davies-Bouldin Index:** 1.0132

## Struktur File

```
├── student-por.csv                          # Dataset utama
├── Tubes_Data_Mining_Kelompok_3_*.ipynb      # Notebook analisis
├── app.py                                   # Dashboard Streamlit
├── requirements.txt                         # Library dependencies
├── student_segment_model_bundle_final.pkl   # Model bundle
├── student_segment_result_final.csv         # Hasil segmentasi
├── cluster_profile_final.csv               # Profil cluster
├── classification_summary.csv              # Evaluasi classification
├── regression_metrics.csv                  # Evaluasi regression
├── cluster_k_evaluation.csv               # Evaluasi k=2-5
└── candidate_feature_evaluation.csv       # Evaluasi kandidat fitur
```

## Menjalankan Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Menu Dashboard

1. **Ringkasan Dashboard** — Penjelasan project, model, dan fitur
2. **Prediksi Siswa** — Input karakteristik → prediksi cluster
3. **Profil Cluster** — Tabel profil dan distribusi
4. **Visualisasi Cluster** — PCA scatter plot
5. **Evaluasi Model** — Clustering, Classification, Regression
6. **Rekomendasi** — Rekomendasi per cluster
