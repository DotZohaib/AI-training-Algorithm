#!/usr/bin/env python3
"""
Streamlit Frontend UI for Student Final Marks Prediction
Run with: streamlit run projects/student_prediction/streamlit_app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Student Marks Predictor", page_icon="🎓")
st.title("🎓 AI Student Final Marks Predictor")
st.write("Enter student academic metrics below to predict final exam marks using our trained Scikit-learn Pipeline.")

study_hours = st.slider("Daily Study Hours", 0.5, 12.0, 6.0, 0.5)
attendance = st.slider("Attendance (%)", 50.0, 100.0, 88.0, 1.0)
prev_score = st.slider("Previous Exam Score", 30.0, 100.0, 75.0, 1.0)
sleep_hours = st.slider("Daily Sleep Hours", 4.0, 10.0, 7.5, 0.5)
practice_tests = st.number_input("Practice Tests Completed", 0, 10, 4)

if st.button("Predict Final Marks"):
    model_path = "models/student_marks_pipeline.pkl"
    pipe = joblib.load(model_path)
    df_in = pd.DataFrame([{
        "study_hours_daily": study_hours,
        "attendance_pct": attendance,
        "previous_score": prev_score,
        "sleep_hours": sleep_hours,
        "practice_tests": practice_tests,
    }])
    pred = round(max(0.0, min(100.0, float(pipe.predict(df_in)[0]))), 2)
    st.success(f"Predicted Final Marks: **{pred} / 100**")
