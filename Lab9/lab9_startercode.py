# lab9_starter.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


# ==========================================================
# DATASETS (PROVIDED - DO NOT MODIFY)
# ==========================================================

def load_dataset_60_40():
    np.random.seed(42)

    normal = np.random.multivariate_normal(
        mean=[50, 50],
        cov=[[200, 80], [80, 200]],
        size=600
    )

    fraud = np.random.multivariate_normal(
        mean=[55, 55],  # VERY close mean
        cov=[[200, 80], [80, 200]],
        size=400
    )

    X = np.vstack([normal, fraud])
    y = np.array([0]*600 + [1]*400)

    return X, y


def load_dataset_90_10():
    np.random.seed(42)

    normal = np.random.multivariate_normal(
        mean=[50, 50],
        cov=[[300, 120], [120, 300]],
        size=900
    )

    fraud = np.random.multivariate_normal(
        mean=[51, 51],
        cov=[[300, 120], [120, 300]],
        size=100
    )

    X = np.vstack([normal, fraud])
    y = np.array([0]*900 + [1]*100)

    return X, y


# ==========================================================
# CORE TASKS
# ==========================================================

def train_logistic(X, y):
    """
    Train logistic regression using cross-entropy loss.
    """
    # i use a fixed random state so results are reproducible
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """
    Return accuracy and F1-score.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return acc, f1



def apply_smote(X, y):
    """
    Apply SMOTE.
    """
    # SMOTE synthesizes new examples for the minority class
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def apply_smoteenn(X, y):
    """
    Apply SMOTEENN.
    """
    # SMOTE ENN combines oversampling (SMOTE) with undersampling (ENN) 
    # to clean up noisy boundaries
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled


def train_weighted_logistic(X, y):
    """
    Train logistic regression with class weights.
    """
    # the 'balanced' argument automatically adjusts weights inversely 
    # proportional to class frequencies. You are penalizing the misclassification of fraud more
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model


# test_lab9.py

import numpy as np
import lab9_startercode as student


def test_dataset_shapes():
    X, y = student.load_dataset_60_40()
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 2


def test_logistic_training():
    X, y = student.load_dataset_60_40()
    model = student.train_logistic(X, y)
    assert hasattr(model, "predict")


def test_metrics_range():
    X, y = student.load_dataset_60_40()
    model = student.train_logistic(X, y)
    acc, f1 = student.evaluate_model(model, X, y)

    assert 0 <= acc <= 1
    assert 0 <= f1 <= 1


def test_smote_increases_data():
    X, y = student.load_dataset_90_10()
    X_s, y_s = student.apply_smote(X, y)

    assert len(X_s) > len(X)


def test_weighted_model_runs():
    X, y = student.load_dataset_90_10()
    model = student.train_weighted_logistic(X, y)
    assert hasattr(model, "predict")



if __name__ == "__main__":
    print("--- Evaluating 90:10 Imbalanced Dataset ---")
    
    # Load the highly imbalanced data
    X, y = load_dataset_90_10()

    # 1. Evaluate Base Model
    base_model = train_logistic(X, y)
    base_acc, base_f1 = evaluate_model(base_model, X, y)
    print(f"Standard Logistic Regression | Accuracy: {base_acc:.4f} | F1-Score: {base_f1:.4f}")

    # 2. Evaluate SMOTE
    print("\n--- Applying SMOTE ---")
    X_smote, y_smote = apply_smote(X, y)
    smote_model = train_logistic(X_smote, y_smote)
    # Evaluating on the original dataset to see how well it generalizes to real imbalance
    smote_acc, smote_f1 = evaluate_model(smote_model, X, y) 
    print(f"SMOTE Logistic Regression    | Accuracy: {smote_acc:.4f} | F1-Score: {smote_f1:.4f}")

    # 3. Evaluate SMOTEENN
    print("\n--- Applying SMOTEENN ---")
    X_smoteenn, y_smoteenn = apply_smoteenn(X, y)
    smoteenn_model = train_logistic(X_smoteenn, y_smoteenn)
    smoteenn_acc, smoteenn_f1 = evaluate_model(smoteenn_model, X, y)
    print(f"SMOTEENN Logistic Regression | Accuracy: {smoteenn_acc:.4f} | F1-Score: {smoteenn_f1:.4f}")

    # 4. Evaluate Cost-Sensitive Model
    print("\n--- Applying Cost-Sensitive Learning ---")
    weighted_model = train_weighted_logistic(X, y)
    weight_acc, weight_f1 = evaluate_model(weighted_model, X, y)
    print(f"Weighted Logistic Regression | Accuracy: {weight_acc:.4f} | F1-Score: {weight_f1:.4f}")