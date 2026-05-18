"""
Lab 13: Student-Side Tests
===========================
Run this file to validate your implementation BEFORE submission.

    pytest test_student.py -v

These tests check that your functions:
    - Exist and return the correct Python types
    - Return dicts / lists with the required keys
    - Produce values in plausible ranges (no NaNs, valid probabilities, etc.)
    - Save the required output image files

These tests do NOT reveal which specific months should be flagged as drifted.
Use the Drift_Summary_Ground_Truth sheet to verify your results yourself.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lab13_solution import (
    DATASET_PATH, FEATURES, TARGET,
    task1_baseline_stats,
    task2_split_batches,
    task3_mean_shift_detection,
    task4_drift_log,
    task5_ks_test,
    task6_method_comparison,
    task7_train_baseline_model,
    task8_evaluate_on_batches,
    task9_drift_dashboard,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def baseline_df():
    return pd.read_excel(DATASET_PATH, sheet_name="Baseline_M1_M3")

@pytest.fixture(scope="session")
def all_df():
    return pd.read_excel(DATASET_PATH, sheet_name="All_Transactions")

@pytest.fixture(scope="session")
def stats(baseline_df):
    return task1_baseline_stats(baseline_df)

@pytest.fixture(scope="session")
def batches(all_df):
    return task2_split_batches(all_df)

@pytest.fixture(scope="session")
def alerts(batches, stats):
    return task3_mean_shift_detection(batches, stats)

@pytest.fixture(scope="session")
def drift_log(batches, stats, alerts):
    return task4_drift_log(batches, stats, alerts)

@pytest.fixture(scope="session")
def ks_results(batches, baseline_df):
    return task5_ks_test(batches, baseline_df)

@pytest.fixture(scope="session")
def comparison(alerts, ks_results):
    return task6_method_comparison(alerts, ks_results)

@pytest.fixture(scope="session")
def model_outputs(baseline_df):
    return task7_train_baseline_model(baseline_df)

@pytest.fixture(scope="session")
def performance(model_outputs, batches):
    model, scaler, _, _ = model_outputs
    return task8_evaluate_on_batches(model, scaler, batches)


# ── Task 1 ────────────────────────────────────────────────────────────────────

class TestTask1:

    def test_returns_dict(self, stats):
        assert isinstance(stats, dict), \
            "task1 must return a dict"

    def test_all_features_present(self, stats):
        for f in FEATURES:
            assert f in stats, f"Missing key '{f}' in stats"

    def test_fraud_rate_key_present(self, stats):
        assert "fraud_rate" in stats, "Missing top-level key 'fraud_rate'"

    def test_subkeys_present_for_each_feature(self, stats):
        required = {"mean", "std", "min", "max"}
        for f in FEATURES:
            missing = required - set(stats[f].keys())
            assert not missing, \
                f"stats['{f}'] is missing keys: {missing}"

    def test_no_nan_in_stats(self, stats):
        for f in FEATURES:
            for k, v in stats[f].items():
                assert not np.isnan(float(v)), \
                    f"NaN found in stats['{f}']['{k}']"

    def test_fraud_rate_is_proportion(self, stats):
        fr = stats["fraud_rate"]
        assert 0.0 <= fr <= 1.0, \
            f"fraud_rate must be in [0, 1], got {fr}"

    def test_std_is_positive(self, stats):
        for f in FEATURES:
            assert stats[f]["std"] > 0, \
                f"stats['{f}']['std'] must be > 0"

    def test_min_leq_mean_leq_max(self, stats):
        for f in FEATURES:
            s = stats[f]
            assert s["min"] <= s["mean"] <= s["max"], \
                f"Violated min <= mean <= max for feature '{f}'"

    def test_transaction_hour_within_day(self, stats):
        s = stats["transaction_hour"]
        assert s["min"] >= 0 and s["max"] <= 23, \
            "transaction_hour values must be in 0-23"

    def test_device_risk_score_bounded(self, stats):
        s = stats["device_risk_score"]
        assert s["min"] >= 0.0 and s["max"] <= 1.0, \
            "device_risk_score must be in [0.0, 1.0]"

    def test_customer_age_plausible(self, stats):
        s = stats["customer_age"]
        assert s["min"] >= 18 and s["max"] <= 80, \
            f"customer_age range [{s['min']}, {s['max']}] looks implausible"


# ── Task 2 ────────────────────────────────────────────────────────────────────

class TestTask2:

    def test_returns_dict(self, batches):
        assert isinstance(batches, dict), "task2 must return a dict"

    def test_exactly_six_batches(self, batches):
        assert len(batches) == 6, \
            f"Expected 6 batches, got {len(batches)}"

    def test_keys_are_ints_1_to_6(self, batches):
        assert set(batches.keys()) == {1, 2, 3, 4, 5, 6}, \
            "Batch keys must be integers 1 through 6"

    def test_each_value_is_dataframe(self, batches):
        for m, df in batches.items():
            assert isinstance(df, pd.DataFrame), \
                f"batches[{m}] must be a pd.DataFrame"

    def test_no_empty_batch(self, batches):
        for m, df in batches.items():
            assert len(df) > 0, f"batches[{m}] is empty"

    def test_total_rows_conserved(self, batches, all_df):
        total = sum(len(df) for df in batches.values())
        assert total == len(all_df), \
            f"Sum of batch rows ({total}) != total rows in all_df ({len(all_df)})"

    def test_each_batch_contains_only_its_month(self, batches):
        for m, df in batches.items():
            assert (df["month"] == m).all(), \
                f"batches[{m}] contains rows from other months"

    def test_feature_columns_present_in_batches(self, batches):
        required = set(FEATURES + [TARGET, "month", "drift_phase"])
        for m, df in batches.items():
            missing = required - set(df.columns)
            assert not missing, \
                f"batches[{m}] missing columns: {missing}"

    def test_plot_file_saved(self):
        assert os.path.exists("task2_batch_means.png"), \
            "task2_batch_means.png not found — call plt.savefig() in task2"


# ── Task 3 ────────────────────────────────────────────────────────────────────

class TestTask3:

    def test_returns_dict(self, alerts):
        assert isinstance(alerts, dict), "task3 must return a dict"

    def test_has_six_entries(self, alerts):
        assert len(alerts) == 6, \
            f"alerts must have 6 entries, got {len(alerts)}"

    def test_keys_are_1_to_6(self, alerts):
        assert set(alerts.keys()) == {1, 2, 3, 4, 5, 6}

    def test_all_values_are_bool(self, alerts):
        for m, v in alerts.items():
            assert isinstance(v, (bool, np.bool_)), \
                f"alerts[{m}] must be bool, got {type(v).__name__}"

    def test_not_all_months_alerted(self, alerts):
        assert not all(alerts.values()), \
            "All 6 months are flagged — threshold logic is likely wrong"

    def test_not_zero_months_alerted(self, alerts):
        assert any(alerts.values()), \
            "No months are flagged at all — threshold logic is likely wrong"

    def test_baseline_months_not_all_alerted(self, alerts):
        assert not all(alerts[m] for m in [1, 2, 3]), \
            "All three baseline months (1-3) are flagged — check your formula"


# ── Task 4 ────────────────────────────────────────────────────────────────────

class TestTask4:

    def test_returns_list(self, drift_log):
        assert isinstance(drift_log, list), "task4 must return a list"

    def test_only_alerted_months_in_log(self, drift_log, alerts):
        alerted = {m for m, v in alerts.items() if v}
        logged  = {e["month"] for e in drift_log}
        assert logged == alerted, \
            f"Logged months {logged} != alerted months {alerted}"

    def test_required_keys_present(self, drift_log):
        required = {"month", "drift_phase", "batch_mean",
                    "shift_magnitude", "sigmas_away"}
        for i, entry in enumerate(drift_log):
            missing = required - entry.keys()
            assert not missing, \
                f"drift_log[{i}] missing keys: {missing}"

    def test_shift_magnitude_non_negative(self, drift_log):
        for e in drift_log:
            assert e["shift_magnitude"] >= 0, \
                f"shift_magnitude must be >= 0, got {e['shift_magnitude']}"

    def test_sigmas_away_gt_2_for_alerted_months(self, drift_log):
        for e in drift_log:
            assert e["sigmas_away"] > 2.0, \
                (f"Month {e['month']}: sigmas_away={e['sigmas_away']:.3f} "
                 f"should be > 2.0 for an alerted month")

    def test_drift_phase_is_valid_string(self, drift_log):
        valid = {"Baseline", "Feature Drift", "Concept Drift"}
        for e in drift_log:
            assert e["drift_phase"] in valid, \
                f"Invalid drift_phase '{e['drift_phase']}'; must be one of {valid}"


# ── Task 5 ────────────────────────────────────────────────────────────────────

class TestTask5:

    def test_returns_dict(self, ks_results):
        assert isinstance(ks_results, dict), "task5 must return a dict"

    def test_has_six_entries(self, ks_results):
        assert len(ks_results) == 6

    def test_required_keys_in_each_entry(self, ks_results):
        required = {"ks_stat", "p_value", "drifted"}
        for m, r in ks_results.items():
            missing = required - r.keys()
            assert not missing, \
                f"ks_results[{m}] missing keys: {missing}"

    def test_ks_stat_in_0_1(self, ks_results):
        for m, r in ks_results.items():
            assert 0.0 <= r["ks_stat"] <= 1.0, \
                f"Month {m}: ks_stat={r['ks_stat']} not in [0, 1]"

    def test_p_value_in_0_1(self, ks_results):
        for m, r in ks_results.items():
            assert 0.0 <= r["p_value"] <= 1.0, \
                f"Month {m}: p_value={r['p_value']} not in [0, 1]"

    def test_drifted_consistent_with_p_value(self, ks_results):
        for m, r in ks_results.items():
            expected = r["p_value"] < 0.05
            assert r["drifted"] == expected, \
                (f"Month {m}: drifted={r['drifted']} but "
                 f"p_value={r['p_value']:.4f} — inconsistent")

    def test_baseline_months_have_high_p_value(self, ks_results):
        for m in [1, 2, 3]:
            assert ks_results[m]["p_value"] > 0.05, \
                (f"Month {m} is a baseline month but p_value="
                 f"{ks_results[m]['p_value']:.4f} — check your implementation")


# ── Task 6 ────────────────────────────────────────────────────────────────────

class TestTask6:

    def test_returns_list(self, comparison):
        assert isinstance(comparison, list), "task6 must return a list"

    def test_has_six_entries(self, comparison):
        assert len(comparison) == 6, \
            f"comparison must have 6 entries, got {len(comparison)}"

    def test_required_keys_present(self, comparison):
        required = {"month", "mean_shift_alert", "ks_alert",
                    "ks_stat", "p_value"}
        for i, e in enumerate(comparison):
            missing = required - e.keys()
            assert not missing, \
                f"comparison[{i}] missing keys: {missing}"

    def test_months_are_1_to_6(self, comparison):
        months = sorted(e["month"] for e in comparison)
        assert months == [1, 2, 3, 4, 5, 6], \
            f"Month values incorrect: {months}"

    def test_mean_shift_alerts_match_task3(self, comparison, alerts):
        for e in comparison:
            m = e["month"]
            assert e["mean_shift_alert"] == alerts[m], \
                f"Month {m}: mean_shift_alert doesn't match task3 output"

    def test_ks_alerts_match_task5(self, comparison, ks_results):
        for e in comparison:
            m = e["month"]
            assert e["ks_alert"] == ks_results[m]["drifted"], \
                f"Month {m}: ks_alert doesn't match task5 output"


# ── Task 7 ────────────────────────────────────────────────────────────────────

class TestTask7:

    def test_returns_four_values(self, model_outputs):
        assert len(model_outputs) == 4, \
            "task7 must return (model, scaler, accuracy, f1)"

    def test_model_is_logistic_regression(self, model_outputs):
        model, _, _, _ = model_outputs
        assert isinstance(model, LogisticRegression), \
            f"model must be LogisticRegression, got {type(model).__name__}"

    def test_scaler_is_standard_scaler(self, model_outputs):
        _, scaler, _, _ = model_outputs
        assert isinstance(scaler, StandardScaler), \
            f"scaler must be StandardScaler, got {type(scaler).__name__}"

    def test_model_is_fitted(self, model_outputs):
        model, _, _, _ = model_outputs
        assert hasattr(model, "coef_"), \
            "Model does not appear to be fitted (missing 'coef_' attribute)"

    def test_scaler_is_fitted(self, model_outputs):
        _, scaler, _, _ = model_outputs
        assert hasattr(scaler, "mean_"), \
            "Scaler does not appear to be fitted (missing 'mean_' attribute)"

    def test_accuracy_in_valid_range(self, model_outputs):
        _, _, accuracy, _ = model_outputs
        assert 0.0 <= accuracy <= 1.0, \
            f"accuracy must be in [0, 1], got {accuracy}"

    def test_f1_in_valid_range(self, model_outputs):
        _, _, _, f1 = model_outputs
        assert 0.0 <= f1 <= 1.0, \
            f"f1 must be in [0, 1], got {f1}"

    def test_model_uses_correct_number_of_features(self, model_outputs):
        model, _, _, _ = model_outputs
        assert model.n_features_in_ == len(FEATURES), \
            (f"Model was trained on {model.n_features_in_} features; "
             f"expected {len(FEATURES)}")

    def test_baseline_accuracy_reasonable(self, model_outputs):
        _, _, accuracy, _ = model_outputs
        assert accuracy >= 0.85, \
            f"Baseline accuracy {accuracy:.4f} seems too low — check your implementation"


# ── Task 8 ────────────────────────────────────────────────────────────────────

class TestTask8:

    def test_returns_dict(self, performance):
        assert isinstance(performance, dict), "task8 must return a dict"

    def test_has_six_entries(self, performance):
        assert len(performance) == 6

    def test_keys_are_1_to_6(self, performance):
        assert set(performance.keys()) == {1, 2, 3, 4, 5, 6}

    def test_required_keys_in_each_entry(self, performance):
        for m, r in performance.items():
            assert "accuracy" in r and "f1" in r, \
                f"performance[{m}] must have 'accuracy' and 'f1'"

    def test_all_scores_in_valid_range(self, performance):
        for m, r in performance.items():
            assert 0.0 <= r["accuracy"] <= 1.0, \
                f"Month {m}: accuracy={r['accuracy']} out of range"
            assert 0.0 <= r["f1"] <= 1.0, \
                f"Month {m}: f1={r['f1']} out of range"

    def test_f1_generally_decreases_after_drift(self, performance):
        avg_baseline = np.mean([performance[m]["f1"] for m in [1, 2, 3]])
        avg_late     = np.mean([performance[m]["f1"] for m in [5, 6]])
        assert avg_late <= avg_baseline + 0.10, \
            ("F1 score should not improve significantly under drift. "
             f"Baseline avg F1={avg_baseline:.4f}, Late avg F1={avg_late:.4f}")


# ── Task 9 ────────────────────────────────────────────────────────────────────

class TestTask9:

    def test_dashboard_file_saved(self, batches, stats, alerts,
                                   ks_results, performance):
        task9_drift_dashboard(batches, stats, alerts, ks_results, performance)
        assert os.path.exists("task9_drift_dashboard.png"), \
            "task9_drift_dashboard.png not found — call plt.savefig() in task9"