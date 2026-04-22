# Diabetes Readmission Predictor — Deployment

Flask API + Docker deployment for the XGBoost readmission prediction model.

---

## Project Structure

```
diabetes_app/
├── app.py                  # Flask application
├── requirements.txt        # Pinned Python dependencies
├── Dockerfile              # Multi-stage production Docker build
├── docker-compose.yml      # Easy local / server deployment
├── models/                 # Place your trained artefacts here
│   ├── xgb_model.pkl
│   ├── scaler_standard.pkl
│   ├── scaler_robust.pkl
│   └── feature_columns.pkl
└── templates/
    ├── index.html          # Prediction web form
    └── result.html         # Result display page
```

---

## Step 1 — Add Your Model Files

Copy the four `.pkl` files saved at the end of the notebook into the `models/` folder:

```bash
mkdir -p models
cp /path/to/xgb_model.pkl          models/
cp /path/to/scaler_standard.pkl    models/
cp /path/to/scaler_robust.pkl      models/
cp /path/to/feature_columns.pkl    models/
```

---

## Step 2 — Run with Docker Compose (recommended)

```bash
# Build image and start container
docker compose up --build

# Run in background
docker compose up --build -d

# Stop
docker compose down
```

The app will be available at **http://localhost:5000**

---

## Step 3 — Run Locally (without Docker)

```bash
pip install -r requirements.txt
python app.py
```

---

## API Endpoints

### `GET /`
Opens the web form for manual patient input.

---

### `POST /predict`
Returns a readmission prediction.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_in_hospital": 5,
    "num_lab_procedures": 45,
    "num_procedures": 2,
    "num_medications": 18,
    "number_diagnoses": 8,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 2,
    "age": 6,
    "insulin": 1,
    "metformin": 1,
    "diabetesMed": 1,
    "change": 0,
    "gender": 0
  }'
```

**Response:**
```json
{
  "label": "Readmitted <30 days",
  "prediction": 1,
  "probability": 0.7231
}
```

---

### `GET /health`
Container health check endpoint.

```json
{ "status": "ok", "model": "xgb_model.pkl" }
```

---

## Notes

- Any features not included in the request body default to `0`.
- The full feature set is loaded from `feature_columns.pkl` at startup.
- Gunicorn runs 4 workers by default. Adjust `--workers` in `Dockerfile` CMD
  based on your server's CPU count: `(2 × cores) + 1` is a common rule.
