"""
Lab 10 Starter Code
Model Optimization & Feature Engineering
"""

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier

# =========================
# LOAD DATA
# =========================
def load_data(path="lab10_data.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


# =========================
# PART A: GRID SEARCH
# =========================
def run_grid_search(X, y):
    """
    TODO:
    - Define param_grid
    - Use GridSearchCV
    - Return best estimator
    """
    """
    Grid Search is exhaustive but expensive. It tests every combination.
    We are tuning max_depth and min_samples_split[cite: 61].
    """
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 20]
    }
    
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_


# =========================
# PART A: RANDOM SEARCH
# =========================
def run_random_search(X, y):
    """
    TODO:
    - Define param_dist
    - Use RandomizedSearchCV
    - Return best estimator
    """
    """
    Random search trades exhaustive certainty for efficiency. 
    It often finds a "good enough" model much faster than Grid Search[cite: 66].
    """
    param_dist = {
        'max_depth': [None] + list(range(2, 30)),
        'min_samples_split': range(2, 30)
    }
    
    model = DecisionTreeClassifier(random_state=42)
    random_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42)
    random_search.fit(X, y)
    
    return random_search.best_estimator_


# =========================
# PART A: OPTUNA
# =========================
def run_optuna(X, y):
    """
    TODO:
    - Define objective
    - Run study.optimize
    - Return best params
    """
    """
    Bayesian Optimization using Optuna. 
    Unlike Grid/Random, it learns from previous trials to make smarter guesses.
    """
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
        
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        return score

    # Suppress optuna logging so it doesn't spam your terminal
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # Running at least 20 trials.
    
    return study.best_params


# =========================
# PART B: CROSS VALIDATION
# =========================
def evaluate_cv(model, X, y):
    """
    Returns:
        mean_accuracy, std_accuracy
    """
    """
    Returning mean accuracy and standard deviation[cite: 70].
    """
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return float(scores.mean()), float(scores.std())


# =========================
# PART C: TARGET ENCODING
# =========================
def target_encode(df, column, target):
    """
    Naive target encoding (leakage-prone)
    """
    """
    Naive target encoding. This is intentionally flawed.
    Encoding the categorical feature using mean target value[cite: 76].
    """
    return df.groupby(column)[target].transform('mean')


def kfold_target_encode(df, column, target, n_splits=5):
    """
    Leakage-safe encoding
    """
    """
    Leakage-safe encoding. 
    We calculate the mean on the training fold to encode the validation fold.
    """
    encoded_col = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
        
        # Calculate means purely on the training set
        means = X_train.groupby(column)[target].mean()
        
        # Map those means to the validation set
        encoded_col.iloc[val_idx] = X_val[column].map(means)
        
    # Fill any NaNs (from categories in val not present in train) with the global mean
    global_mean = df[target].mean()
    encoded_col.fillna(global_mean, inplace=True)
    
    return encoded_col


# =========================
# PART E: CYCLICAL FEATURES
# =========================
def encode_cyclical(feature, max_val):
    """
    Returns:
        sin_feature, cos_feature
    """
    """
    Transforming the hour feature using sine and cosine[cite: 87, 88].
    """
    sin_feature = np.sin(2 * np.pi * feature / max_val)
    cos_feature = np.cos(2 * np.pi * feature / max_val)
    return sin_feature, cos_feature