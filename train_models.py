"""
Train the five required classifiers on the Breast Cancer Wisconsin dataset.
The dataset is bundled in data/breast_cancer_wisconsin.csv, so training does not
require an internet connection.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "breast_cancer_wisconsin.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH)
X = data.drop(columns=["target"])
y = data["target"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
}

rows = []
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    predicted = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted),
    })
    filename = name.lower().replace(" ", "_").replace("-", "")
    joblib.dump(estimator, MODEL_DIR / f"{filename}.joblib")

result = pd.DataFrame(rows)
result.to_csv(ROOT / "model_metrics.csv", index=False)
print(result.to_string(index=False))
print("\\nModels saved in:", MODEL_DIR)
