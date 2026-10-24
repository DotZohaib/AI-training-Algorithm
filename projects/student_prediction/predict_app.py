#!/usr/bin/env python3
"""
Student Final Marks Prediction Application (Inference Script)
Demonstrates how a saved Scikit-learn Pipeline (.pkl) becomes a usable AI application
with input validation and error handling.
"""

import os
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

REQUIRED_FEATURES = [
    "study_hours_daily",
    "attendance_pct",
    "previous_score",
    "sleep_hours",
    "practice_tests",
]


def load_trained_pipeline(model_path="models/student_marks_pipeline.pkl"):
    """Load the saved Scikit-learn pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run 01_ML_Model_Training.ipynb first!"
        )
    return joblib.load(model_path)


def predict_student_marks(student_dict, pipeline):
    """Validate input dictionary and return predicted final marks."""
    missing = [col for col in REQUIRED_FEATURES if col not in student_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df_input = pd.DataFrame([student_dict])[REQUIRED_FEATURES]
    predicted_score = float(pipeline.predict(df_input)[0])
    return round(max(0.0, min(100.0, predicted_score)), 2)


def main():
    # Resolve path whether run from course root or projects/student_prediction/
    model_path = "models/student_marks_pipeline.pkl"
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "..", "models", "student_marks_pipeline.pkl")

    pipeline = load_trained_pipeline(model_path)

    sample_students = [
        {
            "name": "Ali (Consistent Student)",
            "study_hours_daily": 6.5,
            "attendance_pct": 92.0,
            "previous_score": 80.0,
            "sleep_hours": 7.5,
            "practice_tests": 4,
        },
        {
            "name": "Sara (High Achiever)",
            "study_hours_daily": 8.0,
            "attendance_pct": 96.0,
            "previous_score": 89.0,
            "sleep_hours": 7.0,
            "practice_tests": 5,
        },
        {
            "name": "Bilal (Needs Support)",
            "study_hours_daily": 2.0,
            "attendance_pct": 65.0,
            "previous_score": 45.0,
            "sleep_hours": 6.0,
            "practice_tests": 1,
        },
    ]

    print("=== AI Student Marks Predictor (Production Inference) ===")
    for s in sample_students:
        name = s.pop("name")
        pred_marks = predict_student_marks(s, pipeline)
        status = "PASS (Good Standing)" if pred_marks >= 50 else "AT RISK (Extra Coaching Needed)"
        print(f"{name:28s} -> Predicted Marks: {pred_marks:5.2f} / 100 | Status: {status}")


if __name__ == "__main__":
    main()
