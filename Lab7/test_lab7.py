import numpy as np
import pytest

from lab7_starter import (
    compute_distance_matrix,
    linkage_distance,
    find_closest_clusters,
    MyAgglomerative,
    generate_dataset
)


# ==========================================================
# FIXTURE
# ==========================================================

@pytest.fixture
def small_data():
    return np.array([
        [0,0],
        [1,1],
        [5,5],
        [6,6]
    ])


# ==========================================================
# DISTANCE MATRIX
# ==========================================================

def test_distance_matrix(small_data):

    print("\n[TEST] distance matrix")

    D = compute_distance_matrix(small_data)

    assert D.shape == (4,4)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0)


# ==========================================================
# LINKAGE
# ==========================================================

def test_linkage_single(small_data):

    print("\n[TEST] single linkage")

    d = linkage_distance([0], [1], small_data, "single")

    assert d > 0


def test_linkage_methods(small_data):

    print("\n[TEST] all linkage methods")

    for method in ["single", "complete", "average"]:

        d = linkage_distance([0,1], [2,3], small_data, method)

        assert d > 0


# ==========================================================
# FIND CLOSEST
# ==========================================================

def test_find_closest_clusters(small_data):

    print("\n[TEST] closest clusters")

    clusters = [[0], [1], [2], [3]]

    i, j, d = find_closest_clusters(clusters, small_data, "single")

    assert i != j
    assert d >= 0


# ==========================================================
# MODEL FIT
# ==========================================================

def test_fit_runs():

    print("\n[TEST] fit()")

    X = generate_dataset()

    model = MyAgglomerative(n_clusters=3)

    model.fit(X)

    assert model.labels_ is not None


# ==========================================================
# LABELS
# ==========================================================

def test_predict():

    print("\n[TEST] predict()")

    X = generate_dataset()

    model = MyAgglomerative(n_clusters=3)

    model.fit(X)

    labels = model.predict(X)

    assert len(labels) == len(X)