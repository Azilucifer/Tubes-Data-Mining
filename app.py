
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_TITLE = "Dashboard Segmentasi Risiko Belajar Siswa"
BUNDLE_PATH = Path("student_segment_model_bundle_final.pkl")

st.set_page_config(page_title=APP_TITLE, page_icon="\U0001f393", layout="wide")


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        st.error("File student_segment_model_bundle_final.pkl tidak ditemukan.")
        st.stop()
    with open(BUNDLE_PATH, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()

selected_features = bundle["selected_features"]
scaler_kmeans = bundle["scaler_kmeans"]
kmeans_model = bundle["kmeans_model"]
classification_models = bundle["classification_models"]
regression_model = bundle["regression_model"]
cluster_names = bundle["cluster_names"]
cluster_recommendations = bundle["cluster_recommendations"]
cluster_profile = bundle["cluster_profile"]
classification_summary = bundle["classification_summary"]
regression_metrics_df = bundle["regression_metrics"]
clustering_metrics = bundle["clustering_metrics"]

st.title(APP_TITLE)
st.caption(
    "Model utama: K-Means (3 cluster) "
    "| Supervised: Logistic Regression & Gaussian Naive Bayes "
    "| Regression: Linear Regression"
)

menu = st.sidebar.radio(
    "Menu",
    [
        "Ringkasan Dashboard",
        "Prediksi Siswa",
        "Profil Cluster",
        "Visualisasi Cluster",
        "Evaluasi Model",
        "Rekomendasi",
    ],
)

# ================================================================
# 1. RINGKASAN DASHBOARD
# ================================================================
if menu == "Ringkasan Dashboard":
    st.header("Ringkasan Dashboard")

    st.markdown(
        "Dashboard ini merupakan antarmuka interaktif untuk "
        "**Segmentasi Risiko Belajar dan Dukungan Akademik Siswa** "
        "menggunakan dataset `student-por.csv`. Tujuan utama dashboard "
        "adalah menampilkan hasil segmentasi siswa ke dalam tiga kelompok "
        "berdasarkan karakteristik risiko belajar, motivasi pendidikan, "
        "dan dukungan akademik."
    )

    st.subheader("Model yang Digunakan")
    st.markdown(
        '''
| Model | Peran |
|-------|-------|
| **K-Means** (model utama) | Membentuk segmentasi siswa ke dalam 3 cluster berdasarkan kemiripan karakteristik. |
| **Logistic Regression** | Menguji apakah pola cluster K-Means dapat dipelajari oleh model supervised. |
| **Gaussian Naive Bayes** | Pembanding Logistic Regression untuk menguji konsistensi pola cluster. |
| **Linear Regression** | Analisis tambahan untuk memprediksi ketidakhadiran siswa (`absences_capped`). |
'''
    )

    st.subheader("Fitur yang Digunakan")
    feature_table = pd.DataFrame(
        {
            "Variabel": [
                "studytime",
                "failures",
                "absences_capped",
                "higher_num",
                "schoolsup_num",
            ],
            "Makna": [
                "Waktu belajar mingguan siswa.",
                "Riwayat kegagalan akademik sebelumnya.",
                "Jumlah ketidakhadiran setelah capping outlier.",
                "Motivasi melanjutkan pendidikan tinggi.",
                "Dukungan tambahan dari sekolah.",
            ],
        }
    )
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.subheader("Catatan Penting")
    st.warning(
        "**Variabel G1, G2, dan G3 (nilai akademik) tidak digunakan** "
        "dalam seluruh proses analisis, model, dashboard, maupun rekomendasi. "
        "Hasil dashboard tidak boleh dimaknai sebagai prediksi nilai akademik "
        "atau prestasi belajar siswa."
    )
    st.info(
        "Classification hanya menguji konsistensi label cluster, bukan "
        "memprediksi risiko belajar aktual. Regression hanya analisis tambahan. "
        "Rekomendasi utama berasal dari profil cluster K-Means."
    )

# ================================================================
# 2. PREDIKSI SISWA
# ================================================================
elif menu == "Prediksi Siswa":
    st.header("Prediksi Segmentasi Siswa")

    st.markdown(
        "Masukkan karakteristik siswa pada form berikut. Data input akan "
        "diproses menggunakan `StandardScaler` yang sama dengan proses training, "
        "lalu dimasukkan ke model K-Means untuk menentukan cluster segmentasi."
    )

    abs_upper = float(
        bundle.get("absences_capping_bounds", {}).get("upper_bound", 15.0)
    )

    col1, col2 = st.columns(2)

    with col1:
        studytime = st.selectbox(
            "Studytime / waktu belajar mingguan",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "1 \u2014 kurang dari 2 jam/minggu",
                2: "2 \u2014 antara 2 sampai 5 jam/minggu",
                3: "3 \u2014 antara 5 sampai 10 jam/minggu",
                4: "4 \u2014 lebih dari 10 jam/minggu",
            }[x],
        )
        failures = st.selectbox(
            "Failures / kegagalan akademik sebelumnya",
            [0, 1, 2, 3],
            format_func=lambda x: (
                "0 \u2014 Tidak pernah gagal" if x == 0 else f"{x} kali"
            ),
        )
        absences_capped = st.number_input(
            "Absences capped / jumlah ketidakhadiran",
            min_value=0.0,
            max_value=abs_upper,
            value=4.0,
            step=1.0,
        )
        st.caption(
            f"Nilai absensi dibatasi 0\u2013{int(abs_upper)} berdasarkan "
            "hasil capping outlier metode IQR."
        )

    with col2:
        higher_num = st.radio(
            "Ingin melanjutkan pendidikan tinggi?",
            [1, 0],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
        )
        schoolsup_num = st.radio(
            "Mendapat dukungan tambahan dari sekolah?",
            [1, 0],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
        )

    input_df = pd.DataFrame(
        [
            {
                "studytime": studytime,
                "failures": failures,
                "absences_capped": absences_capped,
                "higher_num": higher_num,
                "schoolsup_num": schoolsup_num,
            }
        ]
    )

    if st.button("Prediksi Cluster", use_container_width=True):
        X_input_scaled = scaler_kmeans.transform(input_df[selected_features])
        cluster_result = int(kmeans_model.predict(X_input_scaled)[0])

        st.subheader(
            f"Hasil: Cluster {cluster_result} \u2014 "
            f"{cluster_names.get(cluster_result, 'Tidak dikenal')}"
        )
        st.info(
            cluster_recommendations.get(cluster_result, "Rekomendasi belum tersedia.")
        )

        st.markdown("**Data input siswa:**")
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        st.markdown("**Prediksi supervised learning terhadap cluster:**")
        pred_rows = []
        for model_name, model in classification_models.items():
            pred_cluster = int(model.predict(input_df)[0])
            pred_rows.append(
                {
                    "Model": model_name,
                    "Predicted Cluster": pred_cluster,
                    "Nama Cluster": cluster_names.get(pred_cluster, ""),
                }
            )
        st.dataframe(
            pd.DataFrame(pred_rows), use_container_width=True, hide_index=True
        )

        st.caption(
            "Catatan: Prediksi supervised learning hanya menguji konsistensi "
            "pola cluster K-Means, bukan memprediksi nilai akademik siswa."
        )

# ================================================================
# 3. PROFIL CLUSTER
# ================================================================
elif menu == "Profil Cluster":
    st.header("Profil Cluster")

    st.markdown(
        "Profil cluster dihitung dari **rata-rata setiap fitur** pada "
        "masing-masing cluster hasil K-Means. Interpretasi cluster didasarkan "
        "pada karakteristik dominan yang terlihat dari data, bukan dari "
        "asumsi yang dibuat sebelum analisis."
    )

    st.subheader("Tabel Profil Cluster")
    st.dataframe(cluster_profile, use_container_width=True)

    st.subheader("Distribusi Jumlah Siswa per Cluster")
    if "jumlah_siswa" in cluster_profile.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = cluster_profile["jumlah_siswa"]
        colors_bar = ["#2ecc71", "#3498db", "#e74c3c"]
        bars = ax.bar(
            range(len(counts)),
            counts.values,
            color=colors_bar[: len(counts)],
        )
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"Cluster {i}" for i in counts.index])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Jumlah Siswa")
        ax.set_title("Distribusi Jumlah Siswa per Cluster")
        for bar_item, val in zip(bars, counts.values):
            ax.text(
                bar_item.get_x() + bar_item.get_width() / 2.0,
                bar_item.get_height() + 5,
                str(int(val)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Interpretasi Cluster")
    for cid, cname in cluster_names.items():
        st.markdown(f"**Cluster {cid} \u2014 {cname}**")
        rec = cluster_recommendations.get(cid, "")
        st.markdown(f"> {rec}")

    st.caption(
        "Catatan: Interpretasi cluster tidak menggunakan variabel G1, G2, "
        "atau G3. Nama cluster bersifat deskriptif berdasarkan rata-rata "
        "fitur yang teramati."
    )

# ================================================================
# 4. VISUALISASI CLUSTER
# ================================================================
elif menu == "Visualisasi Cluster":
    st.header("Visualisasi Cluster (PCA)")

    st.markdown(
        "PCA (Principal Component Analysis) digunakan untuk mereduksi lima "
        "fitur menjadi dua komponen utama agar hasil clustering dapat "
        "divisualisasikan dalam scatter plot dua dimensi."
    )

    pca_df = bundle.get("pca_df", None)
    pca_var = bundle.get("pca_explained_variance", None)

    if pca_df is not None:
        if isinstance(pca_df, dict):
            pca_df = pd.DataFrame(pca_df)

        if pca_var is not None:
            pca_var_arr = np.array(pca_var)
            pc1_var = round(float(pca_var_arr[0]) * 100, 2)
            pc2_var = round(float(pca_var_arr[1]) * 100, 2)
            total_var = round(pc1_var + pc2_var, 2)

            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("PC1 Explained Variance", f"{pc1_var}%")
            vc2.metric("PC2 Explained Variance", f"{pc2_var}%")
            vc3.metric("Total Explained Variance", f"{total_var}%")

        fig, ax = plt.subplots(figsize=(8, 6))
        colors_pca = ["#2ecc71", "#3498db", "#e74c3c"]
        for cluster_id in sorted(pca_df["cluster"].unique()):
            mask = pca_df["cluster"] == cluster_id
            label = f"Cluster {cluster_id}: {cluster_names.get(cluster_id, '')}"
            ax.scatter(
                pca_df.loc[mask, "PC1"],
                pca_df.loc[mask, "PC2"],
                s=60,
                alpha=0.7,
                label=label,
                color=colors_pca[int(cluster_id) % len(colors_pca)],
            )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Visualisasi Cluster K-Means Menggunakan PCA")
        ax.legend(title="Cluster", loc="best")
        plt.tight_layout()
        st.pyplot(fig)

        st.info(
            "**Cara membaca grafik PCA:** Setiap titik mewakili satu siswa. "
            "Warna menunjukkan cluster yang ditetapkan K-Means. "
            "Jika cluster terlihat terpisah, pola segmentasi cukup jelas. "
            "Tumpang tindih di beberapa area wajar karena PCA hanya "
            "menampilkan sebagian variasi data."
        )

        st.caption(
            "Catatan: PCA hanya alat bantu visualisasi dua dimensi, bukan "
            "dasar tunggal penentuan kualitas cluster. Keputusan cluster "
            "tetap didasarkan pada metrik evaluasi dan profil cluster."
        )
    else:
        st.warning(
            "Data PCA belum tersedia di model bundle. Jalankan ulang notebook "
            "dan simpan pca_df ke dalam bundle."
        )

# ================================================================
# 5. EVALUASI MODEL
# ================================================================
elif menu == "Evaluasi Model":
    st.header("Evaluasi Model")

    eval_tab = st.radio(
        "Pilih evaluasi:",
        [
            "Evaluasi Clustering K-Means",
            "Evaluasi Classification",
            "Evaluasi Regression",
        ],
        horizontal=True,
    )

    # --- A. EVALUASI CLUSTERING ---
    if eval_tab == "Evaluasi Clustering K-Means":
        st.subheader("Evaluasi Clustering K-Means")

        st.markdown(
            "Evaluasi clustering menggunakan tiga metrik utama untuk menilai "
            "kualitas segmentasi yang dihasilkan K-Means."
        )

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "Silhouette Score",
            f"{clustering_metrics.get('silhouette_score', 0):.4f}",
        )
        mc2.metric(
            "Davies-Bouldin Index",
            f"{clustering_metrics.get('davies_bouldin_index', 0):.4f}",
        )
        mc3.metric(
            "Inertia",
            f"{clustering_metrics.get('inertia', 0):.2f}",
        )

        st.markdown(
            '''
| Metrik | Interpretasi |
|--------|-------------|
| **Silhouette Score** | Mengukur kekompakan dan pemisahan cluster. Rentang -1 sampai 1, semakin tinggi semakin baik. |
| **Davies-Bouldin Index** | Mengukur kualitas pemisahan antarcluster. Semakin kecil semakin baik. |
| **Inertia** | Total jarak data terhadap centroid. Tidak boleh dipakai sendirian untuk memilih k. |
'''
        )

        cluster_eval_data = bundle.get("cluster_eval", None)
        if cluster_eval_data is not None:
            if isinstance(cluster_eval_data, dict):
                cluster_eval_data = pd.DataFrame(cluster_eval_data)

            st.subheader("Perbandingan k = 2 sampai k = 5")
            display_cols = [c for c in cluster_eval_data.columns if c != "distribusi_cluster"]
            st.dataframe(
                cluster_eval_data[display_cols] if len(display_cols) < len(cluster_eval_data.columns) else cluster_eval_data,
                use_container_width=True,
                hide_index=True,
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            k_vals = cluster_eval_data["k"].values

            axes[0].plot(k_vals, cluster_eval_data["inertia"].values, marker="o", color="#2c3e50")
            axes[0].set_title("Elbow Method (Inertia)")
            axes[0].set_xlabel("Jumlah Cluster (k)")
            axes[0].set_ylabel("Inertia")
            axes[0].set_xticks(k_vals)

            axes[1].plot(k_vals, cluster_eval_data["silhouette_score"].values, marker="o", color="#27ae60")
            axes[1].set_title("Silhouette Score per k")
            axes[1].set_xlabel("Jumlah Cluster (k)")
            axes[1].set_ylabel("Silhouette Score")
            axes[1].set_xticks(k_vals)

            axes[2].plot(k_vals, cluster_eval_data["davies_bouldin_index"].values, marker="o", color="#e74c3c")
            axes[2].set_title("Davies-Bouldin Index per k")
            axes[2].set_xlabel("Jumlah Cluster (k)")
            axes[2].set_ylabel("DBI")
            axes[2].set_xticks(k_vals)

            plt.tight_layout()
            st.pyplot(fig)

            st.caption(
                "Pemilihan k mempertimbangkan kombinasi ketiga metrik dan "
                "kemudahan interpretasi cluster, bukan satu metrik saja."
            )

    # --- B. EVALUASI CLASSIFICATION ---
    elif eval_tab == "Evaluasi Classification":
        st.subheader("Evaluasi Classification")

        st.markdown(
            "Classification digunakan untuk menguji apakah pola cluster "
            "K-Means dapat dipelajari oleh model supervised learning. "
            "Target classification berasal dari **label cluster K-Means**, "
            "bukan label asli dari dataset."
        )

        st.dataframe(
            classification_summary, use_container_width=True, hide_index=True
        )

        cm_data = bundle.get("confusion_matrices", None)
        if cm_data is not None:
            st.subheader("Confusion Matrix")
            cm_cols = st.columns(2)

            for idx, (model_name, cm) in enumerate(cm_data.items()):
                cm_array = np.array(cm)
                with cm_cols[idx % 2]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(cm_array, cmap="Blues", aspect="auto")
                    plt.colorbar(im, ax=ax, label="Jumlah")
                    ax.set_title(f"Confusion Matrix\n{model_name}")
                    ax.set_xlabel("Predicted Cluster")
                    ax.set_ylabel("Actual Cluster")
                    ax.set_xticks(range(cm_array.shape[1]))
                    ax.set_yticks(range(cm_array.shape[0]))
                    for i in range(cm_array.shape[0]):
                        for j in range(cm_array.shape[1]):
                            text_color = (
                                "white"
                                if cm_array[i, j] > cm_array.max() / 2
                                else "black"
                            )
                            ax.text(
                                j, i, str(int(cm_array[i, j])),
                                ha="center", va="center", color=text_color,
                            )
                    plt.tight_layout()
                    st.pyplot(fig)

            st.markdown(
                "**Cara membaca confusion matrix:** Nilai pada **diagonal "
                "utama** menunjukkan jumlah prediksi yang benar. Nilai di "
                "**luar diagonal** menunjukkan jumlah prediksi yang salah."
            )

        st.warning(
            "Akurasi classification yang tinggi **tidak berarti** model "
            "mampu memprediksi risiko belajar aktual siswa. Target "
            "classification berasal dari label cluster hasil K-Means, "
            "sehingga akurasi hanya menunjukkan konsistensi pola cluster, "
            "bukan kemampuan prediksi terhadap data baru dengan label asli."
        )

    # --- C. EVALUASI REGRESSION ---
    elif eval_tab == "Evaluasi Regression":
        st.subheader("Evaluasi Regression")

        st.markdown(
            "Linear Regression digunakan sebagai analisis tambahan untuk "
            "memprediksi `absences_capped` menggunakan fitur `studytime`, "
            "`failures`, `higher_num`, dan `schoolsup_num`."
        )

        st.dataframe(
            regression_metrics_df, use_container_width=True, hide_index=True
        )

        reg_pred = bundle.get("regression_actual_pred", None)
        if reg_pred is not None:
            if isinstance(reg_pred, dict):
                reg_pred = pd.DataFrame(reg_pred)

            st.subheader("Scatter Plot Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(
                reg_pred["actual_absences_capped"],
                reg_pred["predicted_absences_capped"],
                alpha=0.7,
                color="#3498db",
            )
            min_val = min(
                reg_pred["actual_absences_capped"].min(),
                reg_pred["predicted_absences_capped"].min(),
            )
            max_val = max(
                reg_pred["actual_absences_capped"].max(),
                reg_pred["predicted_absences_capped"].max(),
            )
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                linestyle="--",
                color="#e74c3c",
                label="Prediksi ideal",
            )
            ax.set_title("Scatter Plot Actual vs Predicted \u2014 Regression")
            ax.set_xlabel("Actual Absences Capped")
            ax.set_ylabel("Predicted Absences Capped")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(
                "**Cara membaca grafik:** Garis diagonal putus-putus "
                "menunjukkan kondisi prediksi ideal (predicted = actual). "
                "Semakin dekat titik-titik ke garis ini, semakin baik "
                "prediksi model. Jika titik menyebar jauh dari garis, "
                "kemampuan model masih terbatas."
            )

        st.warning(
            "Regression hanya analisis tambahan untuk memprediksi "
            "`absences_capped`. Jika R\u00b2 rendah, artinya fitur yang "
            "digunakan belum mampu menjelaskan variasi ketidakhadiran "
            "secara kuat. Rekomendasi utama tetap berdasarkan profil "
            "cluster K-Means, bukan hasil regression."
        )

# ================================================================
# 6. REKOMENDASI
# ================================================================
elif menu == "Rekomendasi":
    st.header("Rekomendasi per Cluster")

    st.markdown(
        "Rekomendasi berikut dibuat berdasarkan **profil rata-rata cluster** "
        "hasil K-Means. Setiap cluster memiliki karakteristik dominan yang "
        "berbeda, sehingga rekomendasi disesuaikan dengan kebutuhan "
        "masing-masing kelompok."
    )

    rec_data = []
    for cid, cname in cluster_names.items():
        rec_data.append(
            {
                "Cluster": cid,
                "Nama Cluster": cname,
                "Rekomendasi": cluster_recommendations.get(cid, ""),
            }
        )
    st.dataframe(
        pd.DataFrame(rec_data), use_container_width=True, hide_index=True
    )

    st.divider()

    for cid, cname in cluster_names.items():
        st.subheader(f"Cluster {cid}: {cname}")
        st.info(
            cluster_recommendations.get(cid, "Rekomendasi belum tersedia.")
        )

    st.caption(
        "Catatan: Rekomendasi tidak berdasarkan nilai G1, G2, dan G3. "
        "Hasil classification hanya menguji konsistensi pola cluster, "
        "bukan memprediksi nilai akademik asli siswa. "
        "Rekomendasi utama dibuat berdasarkan profil cluster K-Means."
    )

# Footer
st.divider()
st.caption(
    "Catatan: G1, G2, dan G3 tidak digunakan dalam model. "
    "Hasil classification hanya menguji konsistensi pola cluster, "
    "bukan memprediksi nilai akademik asli siswa. "
    "Rekomendasi utama dibuat berdasarkan profil cluster K-Means."
)
