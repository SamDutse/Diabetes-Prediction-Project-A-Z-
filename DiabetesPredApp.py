import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app for diabetes prediction
st.title("Diabetes Prediction App")

# Input fields
pregnancies_col, glucose_col, blood_pressure_col = st.columns(3)
pregnancies = pregnancies_col.number_input("Number of Pregnancies", min_value=0)
glucose = glucose_col.number_input("Glucose Level", min_value=0)
blood_pressure = blood_pressure_col.number_input("Blood Pressure", min_value=0)

skin_thickness_col, insulin_col, bmi_col = st.columns(3)
skin_thickness = skin_thickness_col.number_input("Skin Thickness", min_value=0)
insulin = insulin_col.number_input("Insulin", min_value=0)
bmi = bmi_col.number_input("BMI", min_value=0.0)

dpf_col, age_col = st.columns(2)
dpf = dpf_col.number_input("Diabetes Pedigree Function", min_value=0.0)
age = age_col.number_input("Age", min_value=0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.write("The patient is likely to have diabetes.")
    else:
        st.write("The patient is not likely to have diabetes.")
