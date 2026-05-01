import numpy as np
import pytest

from lab6_starter import (
    MyKMeans,
    load_iris_dataset,
    generate_blobs,
    centroid_shift
)

# ==========================================================
# HELPER DATASETS
# ==========================================================

@pytest.fixture
def simple_dataset():
    """
    Small dataset with two obvious clusters.
    """
    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [0.8, 0.9],
        [5.0, 5.0],
        [5.1, 4.9],
        [4.9, 5.2]
    ])
    return X


# ==========================================================
# DATASET LOADING
# ==========================================================

def test_load_iris_dataset():

    X = load_iris_dataset()

    assert isinstance(X, np.ndarray)
    assert X.shape[1] == 2
    assert X.shape[0] == 150


def test_generate_blobs():

    X = generate_blobs(separable=True)

    assert isinstance(X, np.ndarray)
    assert X.shape[1] == 2


# ==========================================================
# DISTANCE METRICS
# ==========================================================

def test_distance_euclidean():

    model = MyKMeans(metric="euclidean")

    x1 = np.array([0,0])
    x2 = np.array([3,4])

    d = model._distance(x1,x2)

    assert pytest.approx(d, rel=1e-3) == 5.0


def test_distance_manhattan():

    model = MyKMeans(metric="manhattan")

    x1 = np.array([0,0])
    x2 = np.array([3,4])

    d = model._distance(x1,x2)

    assert d == 7


def test_distance_cosine():

    model = MyKMeans(metric="cosine")

    x1 = np.array([1,0])
    x2 = np.array([0,1])

    d = model._distance(x1,x2)

    assert pytest.approx(d, rel=1e-3) == 1.0


# ==========================================================
# INITIALIZATION
# ==========================================================

def test_initialize_centroids(simple_dataset):

    model = MyKMeans(k=2)

    centroids = model.initialize_centroids(simple_dataset)

    assert centroids.shape == (2,2)


# ==========================================================
# CLUSTER ASSIGNMENT
# ==========================================================

def test_assign_clusters(simple_dataset):

    model = MyKMeans(k=2)

    centroids = np.array([
        [1,1],
        [5,5]
    ])

    labels = model.assign_clusters(simple_dataset, centroids)

    assert labels.shape[0] == simple_dataset.shape[0]
    assert set(labels) <= {0,1}


# ==========================================================
# CENTROID UPDATE
# ==========================================================

def test_update_centroids(simple_dataset):

    model = MyKMeans(k=2)

    labels = np.array([0,0,0,1,1,1])

    centroids = model.update_centroids(simple_dataset, labels)

    assert centroids.shape == (2,2)


# ==========================================================
# CENTROID SHIFT
# ==========================================================

def test_centroid_shift():

    old = np.array([
        [1,1],
        [5,5]
    ])

    new = np.array([
        [1.1,1],
        [5,5.1]
    ])

    shift = centroid_shift(old,new)

    assert shift > 0


# ==========================================================
# FIT METHOD
# ==========================================================

def test_fit_runs(simple_dataset):

    model = MyKMeans(k=2)

    model.fit(simple_dataset)

    assert model.centroids is not None


# ==========================================================
# PREDICT
# ==========================================================

def test_predict(simple_dataset):

    model = MyKMeans(k=2)

    model.fit(simple_dataset)

    labels = model.predict(simple_dataset)

    assert len(labels) == simple_dataset.shape[0]


# ==========================================================
# INERTIA
# ==========================================================

def test_compute_inertia(simple_dataset):

    model = MyKMeans(k=2)

    model.fit(simple_dataset)

    labels = model.predict(simple_dataset)

    inertia = model.compute_inertia(
        simple_dataset,
        labels,
        model.centroids
    )

    assert inertia >= 0


# ==========================================================
# EMPTY CLUSTER SAFEGUARD
# ==========================================================

def test_empty_cluster_handling():

    X = np.array([
        [1,1],
        [1.1,1],
        [10,10]
    ])

    model = MyKMeans(k=3)

    model.fit(X)

    assert model.centroids.shape == (3,2)