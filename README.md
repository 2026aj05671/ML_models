# Breast Cancer Classification – AIML Assignment 2

## a. Problem statement
Build and compare five supervised classification algorithms for predicting whether a breast tumour observation is benign or malignant. The project includes offline model training, evaluation on a held-out test set, and an interactive Streamlit application that evaluates saved models against an uploaded labelled CSV.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), available through scikit-learn's bundled public dataset.
- **Instances:** 569
- **Predictor features:** 30 numeric measurements computed from digitized images of fine needle aspirates.
- **Target:** `target` (0 = malignant, 1 = benign).
- **Task:** Binary classification.
- **Split:** 80% training / 20% testing, stratified, `random_state=42`.
- **Preprocessing:** StandardScaler is applied within pipelines for Logistic Regression, KNN, and SVM. Decision Tree and Gaussian Naive Bayes use the original numeric features.
- **Data file:** `data/breast_cancer_wisconsin.csv`; held-out test data: `test_data.csv`.

## c. GitHub Repository Link
**Add your repository URL here after creating and pushing the repository.**

## d. Models used

The following table contains the measured results from the fixed stratified hold-out split. AUC uses the probability estimate for class 1 (benign).

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree          |     0.9211 | 0.9163 |      0.9565 |   0.9167 | 0.9362 | 0.8341 |
| K-Nearest Neighbors    |     0.9561 | 0.9788 |      0.9589 |   0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes   |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.9950 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

- **Logistic Regression:** Accuracy 0.982, AUC 0.995, F1 0.986, MCC 0.962. This model achieved the highest F1 score in this run.

- **Decision Tree:** Accuracy 0.921, AUC 0.916, F1 0.936, MCC 0.834. Compare its precision/recall balance with the leading model when selecting a deployment model.

- **K-Nearest Neighbors:** Accuracy 0.956, AUC 0.979, F1 0.966, MCC 0.905. Compare its precision/recall balance with the leading model when selecting a deployment model.

- **Gaussian Naive Bayes:** Accuracy 0.939, AUC 0.988, F1 0.952, MCC 0.868. Compare its precision/recall balance with the leading model when selecting a deployment model.

- **Support Vector Machine:** Accuracy 0.982, AUC 0.995, F1 0.986, MCC 0.962. Compare its precision/recall balance with the leading model when selecting a deployment model.

### Overall winner
**Logistic Regression** achieved the highest F1 score on this particular test split. The final model choice should also consider the cost of false negatives (malignant cases predicted as benign), not only the aggregate score.

## Application features
- Upload a labelled test CSV (`target` plus all 30 feature columns).
- Select one of the five saved classifiers.
- View Accuracy, AUC, Precision, Recall, F1, and MCC.
- View a confusion matrix and classification report.
- View a preview of predictions.
- Models are loaded from `model/`; the app does not retrain on uploaded test data.

## Run locally / in BITS Virtual Lab
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
Upload `test_data.csv` in the app to reproduce the evaluation workflow.

## Repository contents
```text
ML_Assignment_2_Project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── model_metrics.csv
├── test_data.csv
├── data/
│   └── breast_cancer_wisconsin.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Reproducibility and limitations
The test metrics are specific to the stated random split and software versions. Re-run `python train_models.py` in your BITS Virtual Lab and use the resulting `model_metrics.csv` if your environment produces different values. Do not claim the Streamlit app is deployed until you have published it through Streamlit Community Cloud.
