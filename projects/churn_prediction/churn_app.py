#!/usr/bin/env python3
"""
Customer Churn Prediction & Risk Scoring Application (Production Inference Script)
Loads the trained Scikit-learn Pipeline (models/churn_pipeline.pkl) and evaluates
both standard (0.50) and high-recall (0.35) decision thresholds for telecom customers.
"""

import os
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

REQUIRED_FEATURES = [
    "tenure_months",
    "monthly_bill_pkr",
    "support_calls",
    "data_gb_used",
    "contract_months",
]


def load_churn_pipeline(model_path="models/churn_pipeline.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pipeline not found at '{model_path}'. Run 02_Model_Evaluation.ipynb first!"
        )
    return joblib.load(model_path)


def evaluate_customer_risk(customer_dict, pipeline, threshold=0.35):
    missing = [c for c in REQUIRED_FEATURES if c not in customer_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df_in = pd.DataFrame([customer_dict])[REQUIRED_FEATURES]
    prob_churn = float(pipeline.predict_proba(df_in)[0, 1])
    is_churn = int(prob_churn >= threshold)
    risk_band = (
        "HIGH RISK (Immediate Retention Call)"
        if prob_churn >= 0.60
        else "MEDIUM RISK (Send Discount Offer)"
        if prob_churn >= threshold
        else "LOW RISK (Loyal Customer)"
    )
    return round(prob_churn * 100, 1), is_churn, risk_band


def main():
    model_path = "models/churn_pipeline.pkl"
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "..", "models", "churn_pipeline.pkl")

    pipe = load_churn_pipeline(model_path)
    sample_customers = [
        {
            "name": "Usman (New + 6 Support Calls)",
            "tenure_months": 3,
            "monthly_bill_pkr": 5400.0,
            "support_calls": 6,
            "data_gb_used": 18.0,
            "contract_months": 1,
        },
        {
            "name": "Ayesha (3-Year Loyal Contract)",
            "tenure_months": 38,
            "monthly_bill_pkr": 2400.0,
            "support_calls": 1,
            "data_gb_used": 45.0,
            "contract_months": 24,
        },
        {
            "name": "Kamran (Moderate Bill + 4 Calls)",
            "tenure_months": 10,
            "monthly_bill_pkr": 4200.0,
            "support_calls": 4,
            "data_gb_used": 25.0,
            "contract_months": 1,
        },
    ]

    print("=== AI Telecom Customer Churn Risk Engine (Threshold = 0.35) ===")
    for c in sample_customers:
        name = c.pop("name")
        prob_pct, pred_cls, band = evaluate_customer_risk(c, pipe, threshold=0.35)
        print(f"{name:34s} -> Churn Prob: {prob_pct:5.1f}% | Pred: {pred_cls} | Action: {band}")


if __name__ == "__main__":
    main()
