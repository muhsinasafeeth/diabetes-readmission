# 🏥 Diabetes Hospital Readmission Prediction

> A production-quality machine learning pipeline to predict whether a diabetic patient will be readmitted to hospital **within 30 days** of discharge.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Objective](#objective)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [References](#references)

---

## 🔍 Overview

Hospital readmissions within 30 days are a major quality and cost concern in healthcare. Early identification of high-risk diabetic patients allows clinicians to intervene before discharge — potentially reducing unnecessary readmissions and improving patient outcomes.

This project builds a complete, end-to-end ML pipeline on the **UCI Diabetes 130-US Hospitals (1999–2008)** dataset, covering everything from raw data cleaning to hyperparameter-tuned model deployment.

> **Key challenge:** The dataset is heavily imbalanced — only ~11% of patients fall in the positive (readmitted) class.

---

## 🎯 Objective

Build a binary classifier to predict whether a diabetic patient will be readmitted within 30 days (`<30`) of discharge.

| Class | Label | Meaning |
|-------|-------|---------|
| 0 | Not Readmitted | Discharged, no readmission within 30 days |
| 1 | Readmitted | Readmitted within 30 days |

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
- **File:** `diabetic_data.csv`
- **Size:** ~100,000 patient encounter records
- **Features:** 50 attributes including patient demographics, admission details, lab results, diagnosis codes, and medication records
- **Target Variable:** `readmitted` (binarized: `<30` → 1, all others → 0)

---

## 📁 Project Structure

```
├── Mini_Project_code.ipynb     # Main Jupyter notebook (full pipeline)
├── diabetic_data.csv           # Raw dataset (not included — see Dataset section)
├── xgb_model.pkl               # Saved tuned XGBoost model
├── scaler_standard.pkl         # Fitted StandardScaler
├── scaler_robust.pkl           # Fitted RobustScaler
├── feature_columns.pkl         # Ordered feature names for inference
└── README.md
```

---

## 🔬 Methodology

### 1. Data Cleaning
- Replaced `?` placeholders with `NaN`
- Dropped columns with >40% missingness (`weight`, `payer_code`, `medical_specialty`) and non-clinical identifiers
- Imputed remaining missing values with category labels (`'Unknown'`, `'None'`)
- Removed records with expired/hospice discharge codes (not at risk for readmission)
- Binarized the target variable

### 2. Exploratory Data Analysis (EDA)
- Target class distribution (count plot + pie chart)
- Key numeric feature histograms
- Readmission rate by prior inpatient visits
- Correlation heatmap and feature distributions

### 3. Feature Engineering & Encoding

| Encoding Type | Applied To |
|---------------|------------|
| Ordinal | `age` (natural bracket order 0–9) |
| Ordinal | 23 medication columns (`No=0 / Steady=1 / Down=2 / Up=3`) |
| Binary | `gender`, `change`, `diabetesMed` |
| ICD-9 Grouping + One-Hot | `diag_1`, `diag_2`, `diag_3` |
| One-Hot | `race`, `max_glu_serum`, `A1Cresult` |

### 4. Feature Scaling

| Scaler | Applied To | Reason |
|--------|------------|--------|
| `StandardScaler` | `time_in_hospital`, `num_medications`, `number_diagnoses` | Approximately normal distribution |
| `RobustScaler` | `num_lab_procedures`, `num_procedures`, `number_outpatient`, `number_emergency`, `number_inpatient` | Skewed / outlier-prone |

### 5. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training set to balance the ~11% minority class

### 6. Model Building
- Algorithm: **XGBoost Classifier**
- Stratified 80/20 train-test split
- `scale_pos_weight` set to account for class imbalance

### 7. Hyperparameter Tuning

| Step | Method | Purpose |
|------|--------|---------|
| 1 | `RandomizedSearchCV` | Broad search over wide parameter ranges (fast) |
| 2 | `GridSearchCV` | Fine-tune around the best parameters from Step 1 |

- **Scoring metric:** `roc_auc` (most appropriate for imbalanced classification)
- **Cross-validation:** Stratified 5-fold

---

## 📈 Results

### Key Findings

| # | Finding |
|---|---------|
| 1 | **XGBoost** delivered strong ROC-AUC and F1-Score on this imbalanced dataset |
| 2 | **`number_inpatient`** (prior inpatient visits) was the single most influential predictor |
| 3 | **SMOTE** significantly improved recall on the minority class compared to no resampling |
| 4 | Hyperparameter tuning provided incremental gains over an already-competitive baseline |
| 5 | The **Precision-Recall curve** is more informative than ROC given the ~11% class imbalance |

### Evaluation Metrics
Model performance evaluated using: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**, Confusion Matrix, and ROC/Precision-Recall curves.

### Saved Artefacts

| File | Contents |
|------|----------|
| `xgb_model.pkl` | Tuned XGBoost model |
| `scaler_standard.pkl` | StandardScaler (fit on training data only) |
| `scaler_robust.pkl` | RobustScaler (fit on training data only) |
| `feature_columns.pkl` | Ordered feature names required at inference time |

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/diabetes-readmission-prediction.git
   cd diabetes-readmission-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
   ```

3. **Download the dataset**  
   Download `diabetic_data.csv` from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) and place it in the project root directory.

---

## 🚀 Usage

Open and run the notebook end-to-end:

```bash
jupyter notebook Mini_Project_code.ipynb
```

To load the saved model and run a prediction on new data:

```python
import joblib
import pandas as pd

model    = joblib.load('xgb_model.pkl')
features = joblib.load('feature_columns.pkl')

# Prepare your input as a DataFrame with the same feature columns
sample = pd.DataFrame([your_data], columns=features)

prediction  = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print(f"Prediction : {'Readmitted <30d' if prediction == 1 else 'Not Readmitted'}")
print(f"Probability: {probability:.2%}")
```

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualizations
- **Scikit-learn** — preprocessing, model evaluation, hyperparameter tuning
- **XGBoost** — gradient boosted classifier
- **imbalanced-learn (SMOTE)** — class imbalance handling
- **Joblib** — model persistence

---

## 🔮 Future Work

- **Deployment** — serve the saved model via Flask or FastAPI for clinical integration
- **Feature selection** — apply SHAP values for explainability and feature pruning
- **Alternative models** — explore LightGBM, CatBoost, or ensemble stacking
- **Threshold tuning** — optimize the classification threshold for clinical cost-sensitivity

---

## 📚 References

- [UCI Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
- Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates*. BioMed Research International.
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

**Creation :** muhsina v s
**Dataset:** UCI Diabetes 130-US Hospitals (1999–2008)
