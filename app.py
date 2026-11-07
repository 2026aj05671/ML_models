from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="AIML Classification Model Lab", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classification – Model Comparison")
st.write(
    "Upload a labelled test CSV containing the 30 predictor columns and a "
    "`target` column. The app loads saved models and evaluates them; it does "
    "not train or retrain models on uploaded data."
)
st.info("Target encoding: 0 = malignant, 1 = benign.")

uploaded = st.file_uploader("Upload test dataset (CSV)", type=["csv"])
selected_model = st.selectbox("Choose a classification model", list(MODEL_FILES))

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        if "target" not in data.columns:
            st.error("The uploaded CSV must contain a 'target' column.")
            st.stop()

        model_path = MODEL_DIR / MODEL_FILES[selected_model]
        if not model_path.exists():
            st.error(f"Saved model not found: {model_path.name}. Run train_models.py first.")
            st.stop()

        model = joblib.load(model_path)
        X = data.drop(columns=["target"])
        y = data["target"].astype(int)

        expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        if expected is None:
            # Pipelines and estimators trained with a DataFrame generally expose feature_names_in_.
            expected = list(X.columns)
        missing = sorted(set(expected) - set(X.columns))
        extra = sorted(set(X.columns) - set(expected))
        if missing or extra:
            st.error(f"Feature mismatch. Missing columns: {missing}; unexpected columns: {extra}")
            st.stop()

        X = X[expected]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        if y.nunique() < 2:
            st.warning("AUC cannot be calculated because the uploaded target contains only one class.")
            auc = float("nan")
        else:
            auc = roc_auc_score(y, probabilities)

        values = {
            "Accuracy": accuracy_score(y, predictions),
            "AUC": auc,
            "Precision": precision_score(y, predictions, zero_division=0),
            "Recall": recall_score(y, predictions, zero_division=0),
            "F1 Score": f1_score(y, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y, predictions),
        }

        st.subheader(f"Evaluation: {selected_model}")
        cols = st.columns(6)
        for col, (label, value) in zip(cols, values.items()):
            col.metric(label, "N/A" if pd.isna(value) else f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, predictions, labels=[0, 1])
            st.dataframe(pd.DataFrame(
                cm,
                index=["Actual malignant (0)", "Actual benign (1)"],
                columns=["Predicted malignant (0)", "Predicted benign (1)"]
            ), use_container_width=True)
        with right:
            st.subheader("Classification Report")
            report = classification_report(
                y, predictions, labels=[0, 1],
                target_names=["malignant", "benign"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        st.subheader("Prediction preview")
        preview = data.copy()
        preview["predicted_target"] = predictions
        preview["predicted_label"] = preview["predicted_target"].map({0: "malignant", 1: "benign"})
        st.dataframe(preview.head(20), use_container_width=True)
    except Exception as exc:
        st.exception(exc)
else:
    st.caption("For a quick demonstration, upload the project's test_data.csv file.")
