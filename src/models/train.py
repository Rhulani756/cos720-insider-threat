"""
src/models/train.py
-------------------
Hybrid Detection Engine — XGBoost + Isolation Forest.
"""

import os
import sys

# Fix import path — always resolve from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from src.data.preprocess import preprocess, FEATURE_COLS

DATA_PATH      = os.path.join(ROOT, "data", "insider_threat_clean_dataset.csv")
XGB_PATH       = os.path.join(ROOT, "models", "xgboost.pkl")
RF_PATH        = os.path.join(ROOT, "models", "random_forest.pkl")
IF_PATH        = os.path.join(ROOT, "models", "isolation_forest.pkl")
ENCODERS_PATH  = os.path.join(ROOT, "models", "encoders.pkl")
DEPT_STATS_PATH = os.path.join(ROOT, "models", "dept_stats.pkl")

XGB_WEIGHT = 0.6
IF_WEIGHT  = 0.4


def train_xgboost(X_train, y_train):
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Training XGBoost (scale_pos_weight={scale_pos})...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    print("XGBoost trained.")
    return xgb


def train_random_forest(X_train, y_train):
    print("Training Random Forest (for feature importance)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("Random Forest trained.")
    return rf


def train_isolation_forest(X_train):
    print("Training Isolation Forest (unsupervised anomaly detector)...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.054,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    print("Isolation Forest trained.")
    return iso


def compute_composite_score(xgb_model, iso_model, X):
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    iso_raw   = iso_model.score_samples(X)
    iso_min, iso_max = iso_raw.min(), iso_raw.max()
    iso_norm  = 1 - (iso_raw - iso_min) / (iso_max - iso_min + 1e-9)
    composite = (XGB_WEIGHT * xgb_proba + IF_WEIGHT * iso_norm) * 100
    return np.clip(composite, 0, 100)


def cross_validate_model(xgb_model, X_train, y_train):
    print("\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        xgb_model, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        n_jobs=-1,
    )
    print("=== Cross-Validation Results (5-fold) ===")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric.capitalize():10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return cv_results


def evaluate(xgb_model, iso_model, X_test, y_test):
    y_pred    = xgb_model.predict(X_test)
    composite = compute_composite_score(xgb_model, iso_model, X_test)
    cm        = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "confusion_matrix": cm,
        "composite_scores": composite,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }

    print("\n=== Final Test Set Evaluation ===")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  TP:{TP}  FP:{FP}  FN:{FN}  TN:{TN}")
    print(f"  False Positive Rate: {FP/(FP+TN)*100:.2f}%")
    print(f"  False Negative Rate: {FN/(FN+TP)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Malicious"]))
    return metrics


def plot_confusion_matrix(cm, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Malicious"],
                yticklabels=["Normal", "Malicious"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Hybrid Insider Threat Detection")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(rf_model, feature_names, top_n=15, save_path=None):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feature_names[i] for i in indices][::-1],
            importances[indices][::-1], color="#185FA5")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features — Random Forest")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders, dept_stats = preprocess(DATA_PATH)

    xgb_model = train_xgboost(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train)
    iso_model = train_isolation_forest(X_train)

    cross_validate_model(xgb_model, X_train, y_train)
    metrics = evaluate(xgb_model, iso_model, X_test, y_test)

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(xgb_model,  XGB_PATH)
    joblib.dump(rf_model,   RF_PATH)
    joblib.dump(iso_model,  IF_PATH)
    joblib.dump(encoders,   ENCODERS_PATH)
    joblib.dump(dept_stats, DEPT_STATS_PATH)
    joblib.dump(0.65, os.path.join(ROOT, "models", "threshold.pkl"))
    print("\nAll models saved to models/")

    plot_confusion_matrix(metrics["confusion_matrix"],
                          save_path=os.path.join(ROOT, "docs", "confusion_matrix.png"))
    plot_feature_importance(rf_model, FEATURE_COLS,
                            save_path=os.path.join(ROOT, "docs", "feature_importance.png"))