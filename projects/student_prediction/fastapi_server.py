#!/usr/bin/env python3
"""
FastAPI Production Inference Server for Student Final Marks Prediction
Run with: uvicorn projects.student_prediction.fastapi_server:app --reload --port 8000
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Student Marks AI Prediction API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "models/student_marks_pipeline.pkl")
pipeline = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


class StudentInput(BaseModel):
    study_hours_daily: float = Field(..., ge=0.0, le=16.0, description="Daily study hours")
    attendance_pct: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage")
    previous_score: float = Field(..., ge=0.0, le=100.0, description="Previous exam score")
    sleep_hours: float = Field(7.0, ge=3.0, le=12.0, description="Daily sleep hours")
    practice_tests: int = Field(3, ge=0, le=20, description="Practice tests completed")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict_marks(student: StudentInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model file not loaded.")
    df_in = pd.DataFrame([student.dict()])
    pred = float(pipeline.predict(df_in)[0])
    score = round(max(0.0, min(100.0, pred)), 2)
    return {
        "predicted_final_marks": score,
        "status": "PASS" if score >= 50.0 else "AT_RISK",
    }
