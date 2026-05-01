import numpy as np
from lab8_starter import *

def test_dataset():
    X,y = load_fraud_data()
    print("Dataset shape:", X.shape)
    assert X.shape[1] == 2

def test_iforest():
    X,_ = load_fraud_data()
    model = fit_isolation_forest(X)
    labels = predict_iforest(model,X)
    print("IF labels OK")
    assert set(labels).issubset({-1,1})

def test_gaussian_params():
    X,_ = load_fraud_data()
    mu,sigma = compute_gaussian_params(X)
    print("Gaussian params OK")
    assert mu.shape[0] == 2

def test_density():
    X,_ = load_fraud_data()
    mu,sigma = compute_gaussian_params(X)
    d = gaussian_density(X,mu,sigma)
    print("Density OK")
    assert len(d) == len(X)