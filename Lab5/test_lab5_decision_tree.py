import numpy as np
import pytest

import lab5_starter as student


# ============================================================
# TEST DATA
# ============================================================

def simple_dataset():
    """
    Small dataset where correct split is known.
    """

    X = np.array([
        [1],
        [2],
        [3],
        [10],
        [11],
        [12]
    ])

    y = np.array([0,0,0,1,1,1])

    return X, y


# ============================================================
# GINI TESTS
# ============================================================

def test_gini_pure():

    y = np.array([0,0,0,0])

    g = student.gini(y)

    assert abs(g - 0.0) < 1e-6


def test_gini_half():

    y = np.array([0,0,1,1])

    g = student.gini(y)

    assert abs(g - 0.5) < 1e-6


def test_gini_unbalanced():

    y = np.array([0,0,0,1])

    g = student.gini(y)

    assert abs(g - 0.375) < 1e-6


# ============================================================
# INFORMATION GAIN TESTS
# ============================================================

def test_information_gain_positive():

    parent = np.array([0,0,0,1,1,1])

    left = np.array([0,0,0])
    right = np.array([1,1,1])

    gain = student.information_gain(parent, left, right)

    assert gain > 0.4


def test_information_gain_zero():

    parent = np.array([0,1,0,1])

    left = np.array([0,1])
    right = np.array([0,1])

    gain = student.information_gain(parent, left, right)

    assert abs(gain) < 1e-6


# ============================================================
# SPLIT LOGIC TEST
# ============================================================

def test_best_split_simple():

    X, y = simple_dataset()

    feature, threshold, gain = student.find_best_split(X, y)

    assert feature == 0

    # threshold should separate 3 and 10
    assert 3 < threshold < 10

    assert gain > 0.4


# ============================================================
# RETURN TYPES
# ============================================================

def test_return_types():

    X, y = simple_dataset()

    feature, threshold, gain = student.find_best_split(X, y)

    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert isinstance(gain, float)


# ============================================================
# CARDINALITY BIAS TEST
# ============================================================

def test_high_cardinality_feature_bias():

    X, y = simple_dataset()

    # add high-cardinality feature
    index_feature = (np.arange(len(X)) / len(X)).reshape(-1,1)

    X_aug = np.hstack([X, index_feature])

    feature, threshold, gain = student.find_best_split(X_aug, y)

    assert feature in [0,1]  # split should still succeed