import streamlit as st
import numpy as np

# Title
st.title("Diabetes Prediction App")

# Input fields

pregnancies, glucose, blood_pressure = st.columns(3)
pregnancies.number_input("Number of Pregnancies", min_value=0)
glucose.number_input("Glucose Level", min_value=0)
blood_pressure.number_input("Blood Pressure", min_value=0)

skin_thickness, insulin, bmi = st.columns(3)
skin_thickness.number_input("Skin Thickness", min_value=0)
insulin.number_input("Insulin", min_value=0)
bmi.number_input("BMI", min_value=0.0)

dpf, age = st.columns(2)
dpf.number_input("Diabetes Pedigree Function", min_value=0.0)
age.number_input("Age", min_value=0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.write("The patient is likely to have diabetes.")
    else:
        st.write("The patient is not likely to have diabetes.")
