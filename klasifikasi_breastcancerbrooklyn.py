import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Membaca model prediksi anemia
KlasifikasiBreastCancer_model = joblib.load('KlasifikasiBreastCancerBrooklyn_model.sav')

# Membaca model PCA
pca_model = joblib.load('pca_model.sav')

# Membaca model scaler
scaler = joblib.load('scaler_model2.sav')

# Judul web
st.title('Prediksi Diagnosis Kanker Payudara')

# Input pengguna
# form input
radius_mean = st.text_input('Input nilai radius kanker')
texture_mean = st.text_input('Input nilai tekstur kanker')
perimeter_mean = st.text_input('Input nilai Perimeter Kanker')
area_mean = st.text_input('Input nilai Area kanker')
smoothness_mean = st.text_input('Input nilai Smoothness Kanker')
compactness_mean = st.text_input('Input nilai compactness kanker')
concavity_mean = st.text_input('Input nilai Concavity kanker')
concave points_mean = st.text_input('Input nilai Concave Points')
symmetry_mean = st.text_input('Input nilai symmetry kanker')
fractal_dimension_mean = st.text_input('Input nilai fractal dimension')

# Validasi input
if radius_mean_input.strip() and texture_mean_input.strip() and perimeter_mean_input.strip() and area_mean_input.strip() and smoothness_mean_input.strip() and compactness_mean_input.strip() and concavity_mean_input.strip() and concave points_mean_input.strip() and symmetry_mean_input.strip() and fractal_dimension_mean_input.strip():
    radius_mean = float(radius_mean_input)
    texture_mean = float(texture_mean_input)
    perimeter_mean = float(perimeter_mean_input)
    area_mean = float(area_mean_input)
    smoothness_mean = float(smoothness_mean_input)
    compactness_mean = float(compactness_mean_input)
    concavity_mean = float(concavity_mean_input)
    concave points_mean = float(concave points_mean_input)
    symmetry_mean = float(symmetry_mean_input)
    fractal_dimension_mean = float(fractal_dimension_mean_input)

    # Code untuk prediksi
    # Membuat tombol untuk prediksi
    if st.button('Test Prediksi Diagnosis Kanker Payudara'):
        input_data = np.array([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean]).reshape(1, -1)
        breast_prediction = breast_model.predict(input_data)

        # Menampilkan hasil prediksi
        if breast_prediction[0] == 1:
            breast_diagnosis = 'Pasien Diagnosis Kanker Payudara Ganas'
            st.success(breast_diagnosis)
        else:
            breast_diagnosis = 'Pasien Diagnosis Kanker Payudara Jinak'
            st.error(breast_diagnosis)

            # Melakukan clustering untuk penderita anemia
            # Scaling hanya pada variabel yang digunakan untuk pengklasteran
            clustering_data = np.array([RBC_count_in_Millions, HGB_Alltitude_Adjusted, HCT, MCV, MCH, MCHC, RDW]).reshape(1, -1)
            clustering_data_scaled = scaler.transform(clustering_data)

            # Terapkan PCA pada data yang di-scaling
            clustering_data_pca = pca_model.transform(clustering_data_scaled)

            anemia_severity = clustering_model.predict(clustering_data_pca)
            if anemia_severity[0] == 0:
                severity = 'Rendah'
            else:
                severity = 'Tinggi'
            
            st.write(f'Tingkat keparahan anemia: {severity}')
else:
    st.warning('Mohon lengkapi semua kolom input.')
