import numpy as np
import pandas as pd

from lab10_solution import (
    encode_cyclical,
    target_encode,
    evaluate_cv
)

from sklearn.tree import DecisionTreeClassifier


# =========================
# CYCLICAL FEATURES
# =========================
def test_cyclical_identity():
    x = np.array([0, 6, 12, 18])
    sin, cos = encode_cyclical(x, 24)

    # Basic shape
    assert len(sin) == len(x)
    assert len(cos) == len(x)

    # Check known values
    assert np.isclose(sin[0], 0)
    assert np.isclose(cos[0], 1)


def test_cyclical_periodicity():
    x1 = np.array([0])
    x2 = np.array([24])  # same point in cycle

    sin1, cos1 = encode_cyclical(x1, 24)
    sin2, cos2 = encode_cyclical(x2, 24)

    assert np.isclose(sin1, sin2)
    assert np.isclose(cos1, cos2)


# =========================
# TARGET ENCODING
# =========================
def test_target_encoding_correct_means():
    df = pd.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "y": [1, 0, 1, 1]
    })

    encoded = target_encode(df, "cat", "y")

    # A -> mean = 0.5, B -> mean = 1
    assert np.isclose(encoded.iloc[0], 0.5)
    assert np.isclose(encoded.iloc[2], 1.0)


def test_target_encoding_no_missing():
    df = pd.DataFrame({
        "cat": ["A", "B", "C"],
        "y": [1, 0, 1]
    })

    encoded = target_encode(df, "cat", "y")
    assert not encoded.isnull().any()


# =========================
# CROSS VALIDATION
# =========================
def test_evaluate_cv_returns_values():
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, 50)

    model = DecisionTreeClassifier()

    mean, std = evaluate_cv(model, X, y)

    assert isinstance(mean, float)
    assert isinstance(std, float)


def test_evaluate_cv_reasonable_range():
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, 50)

    model = DecisionTreeClassifier()

    mean, std = evaluate_cv(model, X, y)

    assert 0 <= mean <= 1
    assert std >= 0