import streamlit as st
import joblib
import pandas as pd

# Load model dan scaler
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Prediksi Stroke - Decision Tree",
    page_icon="🌳",
    layout="centered"
)

st.title("🌳 Prediksi Risiko Stroke - Decision Tree")

st.write("""
Aplikasi ini digunakan untuk memprediksi risiko stroke berdasarkan data kesehatan pasien
menggunakan model **Decision Tree**.
""")

st.subheader("Input Data Pasien")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Usia", min_value=0.0, max_value=120.0, value=30.0)
hypertension = st.selectbox("Riwayat Hipertensi", ["Tidak", "Ya"])
heart_disease = st.selectbox("Riwayat Penyakit Jantung", ["Tidak", "Ya"])
ever_married = st.selectbox("Status Pernikahan", ["No", "Yes"])
work_type = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Mapping encoding
gender_map = {"Female": 0, "Male": 1}
married_map = {"No": 0, "Yes": 1}
work_map = {
    "Govt_job": 0,
    "Never_worked": 1,
    "Private": 2,
    "Self-employed": 3,
    "children": 4
}
residence_map = {"Rural": 0, "Urban": 1}
smoking_map = {
    "Unknown": 0,
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3
}
yes_no_map = {"Tidak": 0, "Ya": 1}

if st.button("Prediksi Stroke"):
    input_data = pd.DataFrame([[
        gender_map[gender],
        age,
        yes_no_map[hypertension],
        yes_no_map[heart_disease],
        married_map[ever_married],
        work_map[work_type],
        residence_map[residence_type],
        avg_glucose_level,
        bmi,
        smoking_map[smoking_status]
    ]], columns=[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status"
    ])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Hasil Prediksi: Pasien Berisiko Stroke")
    else:
        st.success("Hasil Prediksi: Pasien Tidak Stroke")

st.markdown("---")

st.subheader("Informasi Model Decision Tree")

st.write("Algoritma : Decision Tree")
st.write("Criterion :", model.criterion)
st.write("Max Depth :", model.max_depth)
st.write("Min Samples Split :", model.min_samples_split)
st.write("Min Samples Leaf :", model.min_samples_leaf)

with st.expander("Lihat Detail Model"):
    st.code(str(model))