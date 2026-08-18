"""
ML Assignment 2 — Streamlit App
Dataset: Breast Cancer Wisconsin (Diagnostic)

Features:
  a. CSV upload (test data only, per assignment instructions)
  b. Model selection dropdown
  c. Evaluation metrics display
  d. Confusion matrix / classification report

Run locally:   streamlit run app.py
Deploy:        push this repo to GitHub -> streamlit.io/cloud -> New app -> app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Model Explorer", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    scaler = joblib.load("model/scaler.pkl")
    with open("model/feature_names.json") as f:
        feature_names = json.load(f)
    return models, scaler, feature_names


models, scaler, feature_names = load_artifacts()

st.title("🔬 Classification Model Explorer")
st.caption("Breast Cancer Wisconsin (Diagnostic) dataset — 5 classifiers compared side by side")

# --- a. Dataset upload -------------------------------------------------
st.sidebar.header("1. Upload test data (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv (must include a 'target' column)", type=["csv"]
)

# --- b. Model selection dropdown ---------------------------------------
st.sidebar.header("2. Choose a model")
selected_model_name = st.sidebar.selectbox("Model", list(models.keys()))
selected_model = models[selected_model_name]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must include a 'target' column (the true label).")
        st.stop()

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing expected feature columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    y_true = df["target"]
    X_scaled = scaler.transform(X)

    y_pred = selected_model.predict(X_scaled)
    y_proba = selected_model.predict_proba(X_scaled)[:, 1]

    st.subheader(f"Results — {selected_model_name}")

    # --- c. Evaluation metrics display ---------------------------------
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    col2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
    col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
    col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
    col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
    col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

    # --- d. Confusion matrix / classification report -------------------
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("#### Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

    with st.expander("Compare all 5 models on this uploaded data"):
        rows = []
        for name, model in models.items():
            p = model.predict(X_scaled)
            pr = model.predict_proba(X_scaled)[:, 1]
            rows.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_true, p), 4),
                "AUC": round(roc_auc_score(y_true, pr), 4),
                "Precision": round(precision_score(y_true, p), 4),
                "Recall": round(recall_score(y_true, p), 4),
                "F1": round(f1_score(y_true, p), 4),
                "MCC": round(matthews_corrcoef(y_true, p), 4),
            })
        st.dataframe(pd.DataFrame(rows))

else:
    st.info("👈 Upload the test_data.csv file from the repo to see predictions and metrics.")
    st.markdown(
        "This app loads five pre-trained classifiers (Logistic Regression, Decision Tree, "
        "kNN, Naive Bayes, Random Forest) trained on the Breast Cancer Wisconsin dataset. "
        "Upload the provided `test_data.csv`, pick a model from the sidebar, and view its "
        "metrics, confusion matrix, and classification report."
    )
