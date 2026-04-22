"""
Diabetes Readmission Prediction — Flask API
Loads the trained XGBoost model and exposes two endpoints:
  GET  /          → web form for manual input
  POST /predict   → JSON API for programmatic use
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load artefacts at startup ──────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

model          = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
scaler_std     = joblib.load(os.path.join(MODEL_DIR, "scaler_standard.pkl"))
scaler_rob     = joblib.load(os.path.join(MODEL_DIR, "scaler_robust.pkl"))
feature_cols   = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

STANDARD_COLS = ["time_in_hospital", "num_medications", "number_diagnoses"]
ROBUST_COLS   = ["num_lab_procedures", "num_procedures", "number_outpatient",
                 "number_emergency", "number_inpatient"]


def preprocess(data: dict) -> pd.DataFrame:
    """
    Accept a flat dict of raw feature values and return a scaled
    DataFrame aligned to the training feature columns.
    """
    df = pd.DataFrame([data])

    # Ensure all expected columns exist (fill missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]

    # Apply the same scalers used during training
    df[STANDARD_COLS] = scaler_std.transform(df[STANDARD_COLS])
    df[ROBUST_COLS]   = scaler_rob.transform(df[ROBUST_COLS])

    return df


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the prediction web form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint.

    Accepts:  POST /predict  Content-Type: application/json
    Returns:
      {
        "prediction": 0 | 1,
        "label": "Not Readmitted" | "Readmitted <30 days",
        "probability": 0.0–1.0
      }
    """
    # Accept both JSON body and HTML form submissions
    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()
        # Convert numeric strings to float
        for key in data:
            try:
                data[key] = float(data[key])
            except (ValueError, TypeError):
                pass

    try:
        df      = preprocess(data)
        pred    = int(model.predict(df)[0])
        prob    = float(model.predict_proba(df)[0][1])
        label   = "Readmitted <30 days" if pred == 1 else "Not Readmitted"

        response = {
            "prediction":  pred,
            "label":       label,
            "probability": round(prob, 4),
        }

        # Return JSON for API calls, redirect to result page for form submissions
        if request.is_json:
            return jsonify(response), 200
        return render_template("result.html", **response)

    except Exception as e:
        error = {"error": str(e)}
        if request.is_json:
            return jsonify(error), 400
        return render_template("result.html", error=str(e)), 400


@app.route("/health")
def health():
    """Health check for container orchestration / load balancers."""
    return jsonify({"status": "ok", "model": "xgb_model.pkl"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
