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


# =========================
# DRIVER CODE (The Pipeline with Full Telemetry)
# =========================
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    print("Initializing Data Engineering Pipeline...\n")
    
    try:
        X_raw, y = load_data("lab10_data.csv")
    except FileNotFoundError:
        print("Error: lab10_data.csv missing. Pipeline halted.")
        exit()

    baseline_model = DecisionTreeClassifier(random_state=42)
    df_temp = pd.concat([X_raw, y], axis=1)

    # --- PART D: DATA LEAKAGE COMPARISON ---
    print("--- PART D: Diagnosing Data Leakage ---")
    X_num_only = X_raw.select_dtypes(include=[np.number])
    acc_leaky, _ = evaluate_cv(baseline_model, X_num_only, y)
    print(f"Accuracy BEFORE removing leak: {acc_leaky:.4f}")
    
    leak_tester = DecisionTreeClassifier(random_state=42)
    leak_tester.fit(X_num_only, y)
    for col, imp in zip(X_num_only.columns, leak_tester.feature_importances_):
        if imp > 0.9:
            X_raw.drop(columns=[col], inplace=True)
            print(f"Action: Dropped leaky feature '{col}' (Importance: {imp:.4f})")
            
    X_num_clean = X_raw.select_dtypes(include=[np.number])
    acc_clean, _ = evaluate_cv(baseline_model, X_num_clean, y)
    print(f"Accuracy AFTER removing leak:  {acc_clean:.4f}\n")

    # --- PART C: TARGET ENCODING COMPARISON ---
    print("--- PART C: Target Encoding ---")
    X_naive = X_raw.copy()
    X_kfold = X_raw.copy()
    
    X_naive['city'] = target_encode(df_temp, column='city', target='target')
    X_kfold['city'] = kfold_target_encode(df_temp, column='city', target='target')
    
    acc_naive, _ = evaluate_cv(baseline_model, X_naive, y)
    acc_kfold, _ = evaluate_cv(baseline_model, X_kfold, y)
    
    print(f"Naive Target Encoding CV Accuracy:  {acc_naive:.4f} (Likely inflated)")
    print(f"K-Fold Target Encoding CV Accuracy: {acc_kfold:.4f}")
    
    # Lock in the safe K-Fold encoding for the rest of the pipeline
    X_raw['city'] = X_kfold['city']
    print("Action: Locked in K-Fold Encoding.\n")

    # --- PART E: CYCLICAL FEATURES COMPARISON ---
    print("--- PART E: Cyclical Features ---")
    acc_raw_hour, _ = evaluate_cv(baseline_model, X_raw, y)
    print(f"Before Cyclical Encoding (Raw Hour) CV Accuracy: {acc_raw_hour:.4f}")
    
    X_raw['hour_sin'], X_raw['hour_cos'] = encode_cyclical(X_raw['hour'], max_val=24)
    X_raw.drop(columns=['hour'], inplace=True)
    X = X_raw.copy() # X is now fully clean, encoded, and numeric
    
    acc_cyc_hour, _ = evaluate_cv(baseline_model, X, y)
    print(f"After Cyclical Encoding (Sin/Cos) CV Accuracy:   {acc_cyc_hour:.4f}\n")

    # --- PART B: VALIDATION STRATEGY COMPARISON ---
    print("--- PART B: Validation Strategy ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    baseline_model.fit(X_train, y_train)
    split_acc = accuracy_score(y_test, baseline_model.predict(X_test))
    print(f"Single Train/Test Split Accuracy: {split_acc:.4f}")
    
    mean_acc, std_acc = evaluate_cv(baseline_model, X, y)
    print(f"5-Fold CV Accuracy (Baseline):    {mean_acc:.4f} (+/- {std_acc:.4f})\n")

    # --- PART A: OPTIMIZATION RESULTS ---
    print("--- PART A: Optimization Showdown ---")
    
    print("1. Running Grid Search...")
    best_grid = run_grid_search(X, y)
    grid_acc, _ = evaluate_cv(best_grid, X, y)
    grid_params = best_grid.get_params()
    print(f"   Best Params: max_depth={grid_params['max_depth']}, min_samples_split={grid_params['min_samples_split']}")
    print(f"   -> Accuracy Achieved: {grid_acc:.4f}\n")
    
    print("2. Running Random Search...")
    best_random = run_random_search(X, y)
    random_acc, _ = evaluate_cv(best_random, X, y)
    random_params = best_random.get_params()
    print(f"   Best Params: max_depth={random_params['max_depth']}, min_samples_split={random_params['min_samples_split']}")
    print(f"   -> Accuracy Achieved: {random_acc:.4f}\n")

    print("3. Running Optuna...")
    best_optuna = run_optuna(X, y)
    # Reconstruct the model with Optuna's best params to test it
    optuna_model = DecisionTreeClassifier(max_depth=best_optuna['max_depth'], min_samples_split=best_optuna['min_samples_split'], random_state=42)
    optuna_acc, _ = evaluate_cv(optuna_model, X, y)
    print(f"   Best Params: {best_optuna}")
    print(f"   -> Accuracy Achieved: {optuna_acc:.4f}\n")
    
    print("Pipeline execution finished.")