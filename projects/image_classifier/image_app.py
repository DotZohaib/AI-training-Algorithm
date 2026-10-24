#!/usr/bin/env python3
"""
Industrial Part Defect Classifier (Computer Vision Inference App)
Loads models/vision_defect_classifier.pkl and classifies parts as Normal (0) or Defective (1).
"""

import os
import joblib
import pandas as pd

def main():
    model_path = "models/vision_defect_classifier.pkl"
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "..", "models", "vision_defect_classifier.pkl")
    clf = joblib.load(model_path)

    samples = pd.DataFrame([
        {"edge_density": 0.25, "color_warmth": 0.50, "symmetry_score": 0.88, "contrast_ratio": 0.30},
        {"edge_density": 0.82, "color_warmth": 0.45, "symmetry_score": 0.35, "contrast_ratio": 0.78},
    ])
    preds = clf.predict(samples)
    for i, p in enumerate(preds):
        label = "DEFECTIVE PART (Reject)" if p == 1 else "NORMAL PART (Pass)"
        print(f"Sample Image #{i+1} -> Prediction: {label}")

if __name__ == "__main__":
    main()
