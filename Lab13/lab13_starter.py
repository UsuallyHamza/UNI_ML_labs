"""
Lab 13: Drift Detection in Streaming Data
==========================================
Department of Computer and Software Engineering
SE: Machine Learning

Instructions
------------
- Complete EVERY function marked with # TODO.
- Do NOT rename any function or change its parameters or return types.
- Do NOT remove or reorder the import statements.
- You may add private helper functions anywhere below the imports.
- Run   pytest test_student.py -v   to check your work before submission.

Dataset
-------
File   : transactions_with_drift.xlsx
Sheets :
    All_Transactions           full 6-month dataset  (use for batch splitting)
    Baseline_M1_M3             months 1-3 only        (use as reference)
    Drift_Summary_Ground_Truth per-month true stats   (check AFTER all tasks)

Required packages
-----------------
    pip install pandas numpy scipy scikit-learn openpyxl matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Constants — do not change these
# ------------------------------------------------------------------
DATASET_PATH = "transactions_with_drift.xlsx"
FEATURES     = ["transaction_amount", "customer_age",
                "transaction_hour",   "device_risk_score"]
TARGET       = "is_fraud"


# ==================================================================
# PART A — STREAMING DATA SIMULATION
# ==================================================================

def task1_baseline_stats(baseline_df):
    """
    Task 1: Compute summary statistics for the baseline dataset.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        The Baseline_M1_M3 sheet loaded as a DataFrame.

    Returns
    -------
    stats : dict
        One key per feature in FEATURES, each mapping to a dict with:
            'mean', 'std', 'min', 'max'  (all floats)
        Plus one extra top-level key:
            'fraud_rate' : float   proportion of rows where is_fraud == 1

        Example structure:
        {
            'transaction_amount': {'mean': 2500.0, 'std': 800.0,
                                   'min': 105.3,   'max': 9870.1},
            'customer_age':       { ... },
            'transaction_hour':   { ... },
            'device_risk_score':  { ... },
            'fraud_rate': 0.03
        }
    """
    # TODO: implement
    stats = {}
    return stats


def task2_split_batches(all_df):
    """
    Task 2: Split the full dataset into six monthly batches.

    Parameters
    ----------
    all_df : pd.DataFrame
        The All_Transactions sheet loaded as a DataFrame.

    Returns
    -------
    batches : dict
        Keys   : integer month numbers 1 through 6.
        Values : pd.DataFrame containing only the rows for that month.

    Side effect
    -----------
    Save a figure 'task2_batch_means.png' with two subplots:
        - Top    : mean of transaction_amount per month (line plot)
        - Bottom : mean of transaction_hour per month   (line plot)
    """
    # TODO: implement
    batches = {}
    return batches


# ==================================================================
# PART B — THRESHOLD-BASED DRIFT DETECTION
# ==================================================================

def task3_mean_shift_detection(batches, baseline_stats):
    """
    Task 3: Flag monthly batches using the 2-sigma mean-shift rule.

    For each monthly batch raise a drift alert on transaction_amount if:
        | mean(batch) - mu_baseline | > 2 * sigma_baseline

    Parameters
    ----------
    batches        : dict  — output of task2_split_batches
    baseline_stats : dict  — output of task1_baseline_stats

    Returns
    -------
    alerts : dict
        Keys   : month numbers 1-6.
        Values : bool — True if drift alert triggered, False otherwise.

        Example: {1: False, 2: False, 3: False, 4: False, 5: True, 6: True}
    """
    # TODO: implement
    alerts = {}
    return alerts


def task4_drift_log(batches, baseline_stats, alerts):
    """
    Task 4: Build a structured log of drift events.

    Only include months where alerts[month] is True.

    Parameters
    ----------
    batches        : dict  — output of task2_split_batches
    baseline_stats : dict  — output of task1_baseline_stats
    alerts         : dict  — output of task3_mean_shift_detection

    Returns
    -------
    log : list of dict
        One entry per alerted month. Each dict must have exactly:
            'month'           : int
            'drift_phase'     : str    e.g. 'Feature Drift'
            'batch_mean'      : float  mean of transaction_amount
            'shift_magnitude' : float  | batch_mean - baseline_mean |
            'sigmas_away'     : float  shift_magnitude / baseline_std

    Side effect
    -----------
    Print the log as a readable table using print().
    """
    # TODO: implement
    log = []
    return log


# ==================================================================
# PART C — KS TEST
# ==================================================================

def task5_ks_test(batches, baseline_df):
    """
    Task 5: Apply the KS test to detect distribution shifts.

    Compare the transaction_amount distribution of the baseline
    (Baseline_M1_M3) against each of the 6 monthly batches using
    scipy.stats.ks_2samp.

    Parameters
    ----------
    batches     : dict          — output of task2_split_batches
    baseline_df : pd.DataFrame  — the Baseline_M1_M3 DataFrame

    Returns
    -------
    ks_results : dict
        Keys   : month numbers 1-6.
        Values : dict with keys:
            'ks_stat' : float
            'p_value' : float
            'drifted' : bool   True if p_value < 0.05

        Example:
        {
            1: {'ks_stat': 0.018, 'p_value': 0.923, 'drifted': False},
            5: {'ks_stat': 0.412, 'p_value': 0.000, 'drifted': True},
        }
    """
    # TODO: implement
    ks_results = {}
    return ks_results


def task6_method_comparison(alerts, ks_results):
    """
    Task 6: Side-by-side comparison of the two detection methods.

    Parameters
    ----------
    alerts     : dict — output of task3_mean_shift_detection
    ks_results : dict — output of task5_ks_test

    Returns
    -------
    comparison : list of dict
        One entry per month (1-6). Each dict must have exactly:
            'month'            : int
            'mean_shift_alert' : bool
            'ks_alert'         : bool
            'ks_stat'          : float
            'p_value'          : float

    Side effect
    -----------
    Print the comparison as a readable table using print().
    """
    # TODO: implement
    comparison = []
    return comparison


# ==================================================================
# PART D — CONCEPT DRIFT
# ==================================================================

def task7_train_baseline_model(baseline_df):
    """
    Task 7: Train a Logistic Regression classifier on the baseline data.

    Steps:
        1. Select FEATURES as inputs and TARGET as the label.
        2. Split baseline_df into 80 % train / 20 % test (random_state=42).
        3. Scale features with StandardScaler fitted on the train split only.
        4. Train LogisticRegression(max_iter=1000, random_state=42).
        5. Evaluate on the 20 % test split.

    Parameters
    ----------
    baseline_df : pd.DataFrame — the Baseline_M1_M3 DataFrame

    Returns
    -------
    model    : fitted LogisticRegression
    scaler   : fitted StandardScaler  (needed to transform future batches)
    accuracy : float   on the 20 % baseline test split
    f1       : float   binary F1-score on the 20 % baseline test split
    """
    # TODO: implement
    model    = None
    scaler   = None
    accuracy = 0.0
    f1       = 0.0
    return model, scaler, accuracy, f1


def task8_evaluate_on_batches(model, scaler, batches):
    """
    Task 8: Evaluate the baseline model on each monthly batch.

    Do NOT retrain the model or refit the scaler.
    Apply the same scaler from task7 to each batch before predicting.

    Parameters
    ----------
    model   : fitted LogisticRegression — first output of task7
    scaler  : fitted StandardScaler     — second output of task7
    batches : dict                      — output of task2_split_batches

    Returns
    -------
    performance : dict
        Keys   : month numbers 1-6.
        Values : dict with keys:
            'accuracy' : float
            'f1'       : float
    """
    # TODO: implement
    performance = {}
    return performance


# ==================================================================
# PART E — VISUALIZATION
# ==================================================================

def task9_drift_dashboard(batches, baseline_stats, alerts,
                           ks_results, performance):
    """
    Task 9: Produce a 4-subplot drift dashboard and save it.

    Subplots (top to bottom):
        1. Mean of transaction_amount per month
               Draw horizontal dashed lines at mu ± 2*sigma (alert bounds)
        2. KS statistic per month
               Annotate which months are drifted (p < 0.05)
        3. Fraud rate per month  (mean of is_fraud)
        4. Model F1-score per month (from task8)

    Background shading:
        Months 4-5 : light yellow  (Feature Drift)
        Month  6   : light red     (Concept Drift)

    Parameters
    ----------
    batches        : dict — output of task2_split_batches
    baseline_stats : dict — output of task1_baseline_stats
    alerts         : dict — output of task3_mean_shift_detection
    ks_results     : dict — output of task5_ks_test
    performance    : dict — output of task8_evaluate_on_batches

    Returns
    -------
    None.  Save the figure as 'task9_drift_dashboard.png'.
    """
    # TODO: implement
    pass


# ==================================================================
# MAIN — end-to-end pipeline (runs when you execute this file)
# ==================================================================

if __name__ == "__main__":
    # Load sheets
    baseline_df = pd.read_excel(DATASET_PATH, sheet_name="Baseline_M1_M3")
    all_df      = pd.read_excel(DATASET_PATH, sheet_name="All_Transactions")

    # Part A
    stats   = task1_baseline_stats(baseline_df)
    batches = task2_split_batches(all_df)

    # Part B
    alerts  = task3_mean_shift_detection(batches, stats)
    log     = task4_drift_log(batches, stats, alerts)

    # Part C
    ks_results = task5_ks_test(batches, baseline_df)
    comparison = task6_method_comparison(alerts, ks_results)

    # Part D
    model, scaler, acc, f1 = task7_train_baseline_model(baseline_df)
    performance = task8_evaluate_on_batches(model, scaler, batches)

    # Part E
    task9_drift_dashboard(batches, stats, alerts, ks_results, performance)

    print("\n--- Pipeline complete ---")
    print(f"Baseline model  Accuracy: {acc:.4f}   F1: {f1:.4f}")
    print(f"Mean-shift alerts : {[m for m, v in alerts.items() if v]}")
    print(f"KS-test alerts    : {[m for m, v in ks_results.items() if v['drifted']]}")