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
    
    for feature in FEATURES:
        stats[feature]= {
            'mean': float(baseline_df[feature].mean()), 
            'std' : float(baseline_df[feature].std()),
            'min' : float(baseline_df[feature].min()),
            'max' : float(baseline_df[feature].max())
            }


    stats["fraud_rate"]= baseline_df[TARGET].mean()
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
    months= all_df["month"].unique()

    for m in sorted(months):
        batches[m]= all_df[all_df["month"]==m].copy()

    mean_amounts= []
    mean_hours= []

    for m in sorted(batches.keys()):
        current_month_data= batches[m]

        current_month_amount_mean :float = current_month_data['transaction_amount'].mean()
        current_month_hour_mean :float= current_month_data['transaction_hour'].mean()

        mean_amounts.append(current_month_amount_mean)
        mean_hours.append(current_month_hour_mean)
    

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    
    axes[0].plot(sorted(batches.keys()), mean_amounts, marker='o')
    axes[0].set_title('Mean Transaction Amount per Month')
    axes[0].set_ylabel('Amount (USD)')
    
    axes[1].plot(sorted(batches.keys()), mean_hours, marker='o', color='orange')
    axes[1].set_title('Mean Transaction Hour per Month')
    axes[1].set_ylabel('Hour of Day')
    axes[1].set_xlabel('Month')
    
    plt.tight_layout()
    plt.savefig('task2_batch_means.png')
    plt.close()



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

    base_mean= baseline_stats['transaction_amount']['mean']
    base_std= baseline_stats ['transaction_amount']['std']

    for m, df in batches.items():
        batch_mean= df['transaction_amount'].mean()

        shift_magnitude= abs(batch_mean-base_mean)
        alerts[m]= (shift_magnitude) > 2*(base_std)

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

    base_mean= baseline_stats['transaction_amount']['mean']
    base_std= baseline_stats ['transaction_amount']['std']

    for m, df in batches.items():
        if alerts[m]:
            batch_mean= df['transaction_amount'].mean()
            shift_magnitude= abs(batch_mean-base_mean)

            if m in [4,5]:
                phase= "Feature Drift"
            elif m==6:
                phase= "Concept Drift"
            else:
                phase= "Baseline"

            log.append(
                {
                    "month": m,
                    "drift_phase": phase,
                    "batch_mean": batch_mean,
                    "shift_magnitude": shift_magnitude,
                    "sigmas_away": (shift_magnitude/base_std)
                }
            )
    
    print("\n---Drift log---")
    print(pd.DataFrame(log))
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

    base_data= baseline_df['transaction_amount']

    for m, df in batches.items():
        batch_data= df["transaction_amount"]

        stat, p_value= ks_2samp(base_data, batch_data) #KS test

        ks_results[m]={
            "ks_stat": float(stat),
            "p_value": float(p_value),
            "drifted": p_value < 0.05
        }
    
    
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

    for m in alerts.keys():
        comparison.append({
            "month": m,
            "mean_shift_alert": alerts[m],
            "ks_alert": ks_results[m]["drifted"],
            "ks_stat": ks_results[m]["ks_stat"],
            "p_value": ks_results[m]["p_value"]
        })

    print("\n---Comparison---")
    print(pd.DataFrame(comparison))
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

    #separating feartures and labels
    X= baseline_df[FEATURES]
    y=baseline_df[TARGET]

    # train test data split
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
    
    #scaling of the features
    scaler   = StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled= scaler.fit_transform(X_test)

    #train of model
    model    = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    #evaluation of model
    y_pred=model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)

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
    
    example
    performance={
        1: {
            'accuracy' : 0.5
            'f1'       : 0.7
        }
    }
    """
    # TODO: implement
    performance = {}

    for m, df in batches.items():

        X_batch= df[FEATURES]
        y_batch= df[TARGET]

        X_batch_scaled= scaler.transform(X_batch)
        y_batch_scaled= scaler.transform(y_batch)

        y_pred=model.predict(X_batch_scaled)

        performance[m]={
            "accuracy": accuracy_score(y_batch, y_pred),
            "f1": f1_score(y_batch, y_pred)
        }



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