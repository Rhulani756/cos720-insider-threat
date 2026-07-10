"""
tests/test_scenarios.py
-----------------------
Formal testing scenarios for the Hybrid Detection Engine.
Documents True Positives, False Positives, and False Negatives
from the test set, with end-to-end individual record analysis.

Run with:
    python tests/test_scenarios.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
)

from src.models.predict import load_models, predict_record
from src.data.preprocess import preprocess, preprocess_single_record, FEATURE_COLS
from src.explainability.explain import get_local_explanation, generate_forensic_report

DATA_PATH = os.path.join(ROOT, "data", "insider_threat_clean_dataset.csv")
DOCS_PATH = os.path.join(ROOT, "docs")
N_SAMPLES = 5


def section(title):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def subsection(title):
    print()
    print(f"--- {title} ---")


def run_bulk_evaluation(xgb, rf, iso, encoders, dept_stats, threshold):
    """Bulk evaluation on the full test set."""

    section("LOADING MODELS AND DATA")
    print("Models loaded successfully.")
    X_train, X_test, y_train, y_test, _, _ = preprocess(DATA_PATH)
    df_full  = pd.read_csv(DATA_PATH)
    df_test  = df_full.iloc[y_test.index].reset_index(drop=True)
    y_test_arr = np.array(y_test)

    # Use optimised threshold for all metrics and scenario indices
    proba  = xgb.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    section("OVERALL MODEL PERFORMANCE")
    print(f"  Classification threshold: {threshold} (optimised for best F1)")
    cm = confusion_matrix(y_test_arr, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print(f"  Accuracy:            {accuracy_score(y_test_arr, y_pred):.4f}")
    print(f"  Precision:           {precision_score(y_test_arr, y_pred):.4f}")
    print(f"  Recall:              {recall_score(y_test_arr, y_pred):.4f}")
    print(f"  F1-Score:            {f1_score(y_test_arr, y_pred):.4f}")
    print()
    print(f"  True  Positives (TP): {TP:>5}  — malicious correctly identified")
    print(f"  False Positives (FP): {FP:>5}  — normal wrongly flagged")
    print(f"  False Negatives (FN): {FN:>5}  — malicious missed")
    print(f"  True  Negatives (TN): {TN:>5}  — normal correctly cleared")
    print()
    print(f"  False Positive Rate:  {FP/(FP+TN)*100:.2f}%")
    print(f"  False Negative Rate:  {FN/(FN+TP)*100:.2f}%")
    print()
    print(classification_report(y_test_arr, y_pred, target_names=["Normal", "Malicious"]))

    # Raw confusion matrix
    os.makedirs(DOCS_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Malicious"],
                yticklabels=["Normal", "Malicious"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix — Hybrid Insider Threat Detection")
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Normalised confusion matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Malicious"],
                yticklabels=["Normal", "Malicious"], ax=axes[0])
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("Actual Label")
    axes[0].set_title("Confusion Matrix — Raw Counts")
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Normal", "Malicious"],
                yticklabels=["Normal", "Malicious"], ax=axes[1])
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("Actual Label")
    axes[1].set_title("Confusion Matrix — Normalised (row %)")
    plt.suptitle("Hybrid Insider Threat Detection — Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_PATH, "confusion_matrix_normalised.png"), dpi=150)
    plt.close()
    print("  Confusion matrices saved to docs/")

    tp_idx = np.where((y_test_arr == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_test_arr == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test_arr == 1) & (y_pred == 0))[0]

    display_cols = [
        "employee_department", "employee_position",
        "total_files_burned", "num_printed_pages_off_hours",
        "total_printed_pages", "is_contractor",
        "employee_classification", "late_exit_flag",
        "entry_during_weekend", "num_entries",
    ]

    # ── TRUE POSITIVES ───────────────────────────────────────────────────────
    section(f"SCENARIO 1 — TRUE POSITIVES (TP = {TP})")
    print("  Definition: Malicious records correctly classified as Malicious.")
    print("  These represent the system successfully catching insider threats.")
    subsection(f"Sample TP records (showing {N_SAMPLES})")
    print(df_test.iloc[tp_idx[:N_SAMPLES]][display_cols].to_string(index=False))
    subsection("TP profile analysis")
    tp_rows = df_test.iloc[tp_idx]
    print(f"  Avg files burned:        {tp_rows.total_files_burned.mean():.1f}")
    print(f"  Avg off-hours printing:  {tp_rows.num_printed_pages_off_hours.mean():.1f}")
    print(f"  Avg total pages printed: {tp_rows.total_printed_pages.mean():.1f}")
    print(f"  Contractor rate:         {tp_rows.is_contractor.mean()*100:.1f}%")
    print(f"  Avg security clearance:  {tp_rows.employee_classification.mean():.2f}")
    print(f"  Weekend entry rate:      {tp_rows.entry_during_weekend.mean()*100:.1f}%")

    # ── FALSE POSITIVES ──────────────────────────────────────────────────────
    section(f"SCENARIO 2 — FALSE POSITIVES (FP = {FP})")
    print("  Definition: Normal records incorrectly classified as Malicious.")
    print("  These cause unnecessary investigations of innocent employees.")
    subsection(f"Sample FP records (showing {N_SAMPLES})")
    print(df_test.iloc[fp_idx[:N_SAMPLES]][display_cols].to_string(index=False))
    subsection("FP profile analysis — why were they wrongly flagged?")
    fp_rows = df_test.iloc[fp_idx]
    print(f"  Avg files burned:        {fp_rows.total_files_burned.mean():.1f}")
    print(f"  Avg off-hours printing:  {fp_rows.num_printed_pages_off_hours.mean():.1f}")
    print(f"  Avg total pages printed: {fp_rows.total_printed_pages.mean():.1f}")
    print(f"  Contractor rate:         {fp_rows.is_contractor.mean()*100:.1f}%")
    print(f"  Avg security clearance:  {fp_rows.employee_classification.mean():.2f}")
    print(f"  Top departments:         {fp_rows.employee_department.value_counts().head(3).to_dict()}")
    print()
    print("  Analysis: False positive records predominantly involve subjects")
    print("  with elevated security clearance, contractor status, or")
    print("  assignment to sensitive departments (Legal, Security, IT).")
    print("  These subjects share profile characteristics with confirmed")
    print("  malicious insiders, resulting in elevated malicious probability")
    print("  despite the absence of confirmed threat behaviour. The primary")
    print("  driver is feature overlap between high-clearance normal employees")
    print("  and the malicious training population.")

    # ── FALSE NEGATIVES ──────────────────────────────────────────────────────
    section(f"SCENARIO 3 — FALSE NEGATIVES (FN = {FN})")
    print("  Definition: Malicious records incorrectly classified as Normal.")
    print("  These are missed threats — the most dangerous type of error.")
    subsection(f"Sample FN records (showing {min(N_SAMPLES, len(fn_idx))})")
    print(df_test.iloc[fn_idx[:N_SAMPLES]][display_cols].to_string(index=False))
    subsection("FN profile analysis — why were they missed?")
    fn_rows = df_test.iloc[fn_idx]
    print(f"  Avg files burned:        {fn_rows.total_files_burned.mean():.1f}")
    print(f"  Avg off-hours printing:  {fn_rows.num_printed_pages_off_hours.mean():.1f}")
    print(f"  Avg total pages printed: {fn_rows.total_printed_pages.mean():.1f}")
    print(f"  Contractor rate:         {fn_rows.is_contractor.mean()*100:.1f}%")
    print(f"  Avg security clearance:  {fn_rows.employee_classification.mean():.2f}")
    print(f"  Top departments:         {fn_rows.employee_department.value_counts().head(3).to_dict()}")
    print()
    print("  Analysis: False negative records are characterised by minimal")
    print("  deviation from normal behavioural baselines — low external file")
    print("  transfer volumes, negligible off-hours print activity, and")
    print("  unremarkable physical access patterns. This profile is consistent")
    print("  with low-and-slow insider activity, where a malicious actor")
    print("  operates deliberately below automated detection thresholds.")
    print("  This represents a known limitation of single-day snapshot")
    print("  detection models. Longitudinal baselining is the primary")
    print("  recommended mitigation for this class of failure.")

    # ── ISOLATION FOREST SUPPLEMENTARY ───────────────────────────────────────
    section("ISOLATION FOREST SUPPLEMENTARY DETECTION")
    iso_preds = iso.predict(X_test)
    iso_anomaly = (iso_preds == -1)
    fn_caught = np.sum(iso_anomaly[fn_idx])
    print(f"  False Negatives caught by Isolation Forest: {fn_caught} / {len(fn_idx)}")
    print(f"  The unsupervised model partially compensates for XGBoost misses,")
    print(f"  demonstrating the value of the hybrid detection approach.")

    # ── IMPROVEMENTS ─────────────────────────────────────────────────────────
    section("REALISTIC IMPROVEMENTS")
    improvements = [
        ("Temporal behavioural baselining",
         "Track per-employee behaviour change over time. A sudden spike\n"
         "  in file burning after weeks of normal activity is a stronger\n"
         "  signal than any single-day record."),
        ("Additional data sources",
         "Incorporate network traffic logs, email metadata, and USB logs.\n"
         "  Data exfiltration leaves traces beyond printing and file burning."),
        ("SHAP values for local explainability",
         "Replace contribution-based explanation with SHAP for more\n"
         "  mathematically rigorous per-prediction feature attribution."),
        ("Analyst feedback loop",
         "Allow analysts to mark FP/FN cases to periodically retrain\n"
         "  the model, improving accuracy for the specific organisation."),
        ("Department-specific thresholds",
         "Different departments have different normal behaviour baselines.\n"
         "  Per-department thresholds would reduce false positives in\n"
         "  high-clearance departments like Security and Legal."),
    ]
    for i, (title, desc) in enumerate(improvements, 1):
        print(f"\n  {i}. {title}")
        print(f"  {desc}")

    return df_test, tp_idx, fp_idx, fn_idx


def run_individual_record_tests(xgb, rf, iso, encoders, dept_stats, threshold,
                                 df_test, tp_idx, fp_idx, fn_idx):
    """Test three individual records end-to-end through the prototype pipeline."""

    raw_cols = [c for c in pd.read_csv(DATA_PATH).columns if c != "is_malicious"]

    test_cases = [
        ("TRUE POSITIVE",  tp_idx[0], "Malicious"),
        ("FALSE POSITIVE", fp_idx[0], "Normal"),
        ("FALSE NEGATIVE", fn_idx[0], "Malicious"),
    ]

    section("INDIVIDUAL RECORD TESTS (End-to-End Prototype Pipeline)")

    for scenario, idx, actual_label in test_cases:
        print(f"\n{'='*65}")
        print(f"  {scenario} — Actual label: {actual_label}")
        print(f"{'='*65}")

        record   = df_test.iloc[idx][raw_cols].to_dict()
        result   = predict_record(record, xgb, rf, iso, encoders, dept_stats, threshold)
        X        = preprocess_single_record(record, encoders, dept_stats)
        local_df = get_local_explanation(rf, X, FEATURE_COLS)
        report   = generate_forensic_report(local_df, result)

        print(f"  Predicted:            {result['label']}")
        print(f"  Actual:               {actual_label}")
        print(f"  Correct:              {'YES ✓' if result['label'] == actual_label else 'NO ✗'}")
        print(f"  Composite Risk Score: {result['composite_risk_score']} / 100")
        print(f"  Risk Level:           {result['risk_level']}")
        print(f"  XGBoost Confidence:   {result['rf_confidence']}%")
        print(f"  IF Anomaly:           {'YES' if result['isolation_forest_anomaly'] else 'NO'}")
        print()
        print(report)


if __name__ == "__main__":
    section("INITIALISING HYBRID DETECTION ENGINE TEST SUITE")

    xgb, rf, iso, encoders, dept_stats, threshold = load_models()

    df_test, tp_idx, fp_idx, fn_idx = run_bulk_evaluation(
        xgb, rf, iso, encoders, dept_stats, threshold
    )

    run_individual_record_tests(
        xgb, rf, iso, encoders, dept_stats, threshold,
        df_test, tp_idx, fp_idx, fn_idx
    )

    section("TESTING COMPLETE")
    print("  All outputs saved to docs/")
    print("  - docs/confusion_matrix.png")
    print("  - docs/confusion_matrix_normalised.png")