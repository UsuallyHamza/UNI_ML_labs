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



