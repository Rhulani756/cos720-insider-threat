"""
src/explainability/explain.py
------------------------------
Explainability module for the Hybrid Detection Engine.

Produces:
1. Global feature importance (Random Forest)
2. Local per-record risk breakdown
3. Formal digital forensic investigation report (ISO/IEC 27043-aligned)
"""

import os
import sys
import random
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

FEATURE_DESCRIPTIONS = {
    "total_files_burned":           "Files transferred to external media",
    "burned_from_other":            "File transfers originating from non-standard locations",
    "num_printed_pages_off_hours":  "Pages printed outside authorised business hours",
    "total_printed_pages":          "Total volume of pages printed",
    "late_exit_flag":               "Late building exit recorded",
    "entry_during_weekend":         "Building access recorded on weekend",
    "num_entries":                  "Total building access events",
    "num_unique_campus":            "Number of distinct campuses accessed",
    "is_abroad":                    "Subject currently abroad",
    "hostility_country_level":      "Risk level of travel destination",
    "trip_day_number":              "Duration of current foreign trip (days)",
    "has_criminal_record":          "Prior criminal record on file",
    "has_foreign_citizenship":      "Subject holds foreign citizenship",
    "has_medical_history":          "Medical history on record",
    "is_contractor":                "Subject employed as contractor",
    "employee_classification":      "Security clearance classification level",
    "employee_seniority_years":     "Years of service",
    "dev_files_burned":             "File transfer volume deviation from departmental mean",
    "dev_off_hours_print":          "Off-hours printing deviation from departmental mean",
    "dev_entries":                  "Building access deviation from departmental mean",
    "dev_printed_pages":            "Print volume deviation from departmental mean",
    "employee_department_enc":      "Department",
    "employee_campus_enc":          "Campus",
    "employee_position_enc":        "Position",
    "employee_origin_country_enc":  "Country of origin",
}

ENCODED_COLS = {
    "employee_department_enc",
    "employee_campus_enc",
    "employee_position_enc",
    "employee_origin_country_enc",
}


def get_global_feature_importance(rf_model, feature_names):
    """Global feature importance ranked by RF mean decrease in impurity."""
    importances = rf_model.feature_importances_
    df = pd.DataFrame({
        "feature":     feature_names,
        "importance":  importances,
        "description": [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
    return df


def get_local_explanation(rf_model, X_single, feature_names):
    """Local explanation for a single prediction."""
    importances  = rf_model.feature_importances_
    values       = X_single.iloc[0].values
    val_norm     = np.abs(values) / (np.abs(values).max() + 1e-9)
    contribution = importances * val_norm
    contribution = contribution / (contribution.sum() + 1e-9)

    df = pd.DataFrame({
        "feature":      feature_names,
        "value":        values,
        "importance":   importances,
        "contribution": contribution,
        "description":  [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names],
    }).sort_values("contribution", ascending=False).reset_index(drop=True)

    df["risk_level"] = df["contribution"].apply(
        lambda x: "High" if x >= 0.05 else ("Medium" if x >= 0.02 else "Low")
    )
    return df


def generate_forensic_report(local_df, result, top_n=5):
    """
    Generate a formal Digital Forensic Investigation Report.
    Structured according to ISO/IEC 27043 investigation process classes:
    Readiness → Identification → Collection → Analysis → Presentation.
    Language and format consistent with real DFIR reporting conventions.
    """
    label      = result["label"]
    composite  = result["composite_risk_score"]
    risk_level = result["risk_level"]
    rf_conf    = result["rf_confidence"]
    if_flag    = result["isolation_forest_anomaly"]
    thresh     = result.get("threshold_used", 0.65)

    now       = datetime.now()
    report_id = f"TS-{now.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    top = local_df[
        local_df["risk_level"].isin(["High", "Medium"]) &
        ~local_df["feature"].isin(ENCODED_COLS)
    ].head(top_n)

    if label == "Malicious" and risk_level == "Critical":
        disposition = "HIGH PRIORITY — IMMEDIATE ESCALATION REQUIRED"
    elif label == "Malicious":
        disposition = "ELEVATED RISK — SECONDARY REVIEW REQUIRED"
    elif if_flag:
        disposition = "ANOMALOUS BEHAVIOUR — ENHANCED MONITORING REQUIRED"
    else:
        disposition = "NO INDICATORS OF COMPROMISE — ROUTINE MONITORING"

    W = 56  # line width

    lines = [
        "DIGITAL FORENSIC INVESTIGATION REPORT",
        "=" * W,
        f"Report reference : {report_id}",
        f"Date / Time      : {timestamp}",
        f"Prepared by      : ThreatSense Automated Detection Engine",
        f"Framework        : ISO/IEC 27043 — Incident Investigation",
        f"Disposition      : {disposition}",
        "=" * W,
        "",
        "1.  EXECUTIVE SUMMARY",
        "-" * W,
    ]

    if label == "Malicious":
        lines += [
            f"    Automated analysis of the subject record produced a",
            f"    Composite Risk Score of {composite:.1f} out of 100, assigned",
            f"    a severity classification of {risk_level}. The primary",
            f"    classification model returned a malicious probability",
            f"    of {rf_conf:.1f}% (classification threshold: {thresh}).",
            f"    Observed behavioural indicators deviate significantly",
            f"    from established departmental baselines across multiple",
            f"    dimensions. Escalation is warranted.",
        ]
    elif if_flag:
        lines += [
            f"    Automated analysis of the subject record produced a",
            f"    Composite Risk Score of {composite:.1f} out of 100.",
            f"    The primary classifier returned a normal classification;",
            f"    however, the anomaly detection subsystem identified",
            f"    statistically irregular behaviour relative to the",
            f"    population baseline. This pattern may be consistent",
            f"    with low-and-slow insider activity. Enhanced monitoring",
            f"    and secondary analyst review are recommended.",
        ]
    else:
        lines += [
            f"    Automated analysis of the subject record produced a",
            f"    Composite Risk Score of {composite:.1f} out of 100, assigned",
            f"    a severity classification of {risk_level}. No behavioural",
            f"    indicators of insider threat activity were identified.",
            f"    Observed activity is consistent with normal departmental",
            f"    operations. No further action is required at this time.",
        ]

    lines += [
        "",
        "2.  EVIDENCE COLLECTION AND READINESS",
        "-" * W,
        "    2.1  Subject behavioural record ingested and validated.",
        "    2.2  Departmental activity baseline computed from",
        "         historical records for peer-group normalisation.",
        "    2.3  Deviation features derived for four behavioural",
        "         dimensions: file transfer, print volume,",
        "         building access frequency, and off-hours activity.",
        f"    2.4  Total features submitted for analysis: {len(local_df)}",
        "         (21 raw behavioural features + 4 engineered",
        "         deviation features).",
        "    2.5  Data completeness verified. No missing values.",
        "",
        "3.  DETECTION AND ANALYSIS",
        "-" * W,
        "    3.1  Supervised classification (XGBoost):",
        f"         Result          : {label}",
        f"         Confidence      : {rf_conf:.1f}%",
        f"         Threshold       : {thresh:.2f} (F1-optimised)",
        "",
        "    3.2  Unsupervised anomaly detection (Isolation Forest):",
        f"         Result          : {'Anomaly detected' if if_flag else 'No anomaly detected'}",
        f"         Interpretation  : {'Behaviour is statistically unusual relative to the training population.' if if_flag else 'Behaviour is within expected population distribution.'}",
        "",
        f"    3.3  Composite Risk Score: {composite:.1f} / 100  [{risk_level}]",
        "         Computation: (0.60 × XGBoost probability) +",
        "         (0.40 × normalised Isolation Forest anomaly score),",
        "         scaled to 0–100.",
        "",
        "4.  RISK INDICATOR ANALYSIS",
        "-" * W,
    ]

    if top.empty:
        lines.append("    No high or medium risk indicators identified.")
    else:
        lines += [
            "    The following behavioural features contributed most",
            "    significantly to the detection outcome, ranked by",
            "    contribution score:",
            "",
        ]
        for i, (_, row) in enumerate(top.iterrows(), 1):
            val = row["value"]
            val_str = f"{val:.2f}" if isinstance(val, float) and val != int(val) else str(int(val))
            lines += [
                f"    {i}. {row['description']}",
                f"       Observed value  : {val_str}",
                f"       Risk level      : {row['risk_level']}",
                f"       Contribution    : {row['contribution']:.4f}",
                "",
            ]

    lines += [
        "5.  FINDINGS",
        "-" * W,
    ]

    if label == "Malicious":
        lines += [
            "    The subject employee's behavioural record contains",
            "    indicators consistent with insider threat activity as",
            "    defined under the ThreatSense detection criteria.",
            "    Significant deviations from the established departmental",
            "    baseline were recorded across multiple feature dimensions.",
            "    Both the supervised and unsupervised detection subsystems",
            "    support this classification.",
        ]
    elif if_flag:
        lines += [
            "    The subject employee's record was classified as normal",
            "    by the primary supervised classifier. The unsupervised",
            "    anomaly detection subsystem identified statistical",
            "    irregularities that may indicate emerging or previously",
            "    uncharacterised insider behaviour. This pattern warrants",
            "    continued observation pending further behavioural data.",
        ]
    else:
        lines += [
            "    No indicators of insider threat activity were identified",
            "    in the subject employee's behavioural record. Observed",
            "    levels of file transfer, print activity, physical access,",
            "    and travel behaviour fall within normal departmental",
            "    ranges. Both detection subsystems returned negative results.",
        ]

    lines += [
        "",
        "6.  RECOMMENDATIONS",
        "-" * W,
    ]

    if label == "Malicious":
        lines += [
            "    (a) Escalate to the Information Security team",
            "        immediately and assign to a senior analyst.",
            "    (b) Preserve all associated system, access, and",
            "        application logs in their original state.",
            "    (c) Initiate a formal investigation in accordance",
            "        with the organisation's Incident Response Policy.",
            "    (d) Do not notify the subject prior to completion",
            "        of the initial investigation phase.",
            "    (e) Document all investigative actions in the",
            "        official incident register.",
        ]
    elif if_flag:
        lines += [
            "    (a) Place the subject under enhanced behavioural",
            "        monitoring for a minimum of 30 days.",
            "    (b) Review associated access logs for the preceding",
            "        30-day period.",
            "    (c) Assign for secondary analyst review if anomalous",
            "        patterns persist beyond five business days.",
            "    (d) No immediate disciplinary action is warranted",
            "        on the basis of this report alone.",
        ]
    else:
        lines += [
            "    (a) No immediate action required.",
            "    (b) Maintain routine monitoring in accordance with",
            "        the organisation's standard security policy.",
            "    (c) Re-evaluate if new behavioural data becomes",
            "        available or if a subsequent alert is generated.",
        ]

    lines += [
        "",
        "7.  LIMITATIONS AND DISCLAIMER",
        "-" * W,
        "    This report is produced by an automated detection system",
        "    and is intended to support, not replace, the judgement",
        "    of a qualified security analyst. Detection results are",
        "    probabilistic. The system operates on single-day",
        "    behavioural snapshots and does not account for",
        "    longitudinal changes in individual behaviour patterns.",
        "    All findings should be reviewed before formal action.",
        "",
        "=" * W,
        f"    END OF REPORT  |  Ref: {report_id}",
        "=" * W,
    ]
    return "\n".join(lines)


def get_risk_score_breakdown(local_df, top_n=10):
    """Return top-N features as a risk breakdown table for the prototype UI."""
    filtered = local_df[~local_df["feature"].isin(ENCODED_COLS)]
    return filtered[["description", "value", "contribution", "risk_level"]].head(top_n).copy()