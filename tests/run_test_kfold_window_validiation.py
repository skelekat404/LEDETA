# run_rq_kfold_window_validation.py
# PURPOSE: Validates that email cases are not leaking across multiple 30-day windows
# by running 80/20 stratified train/test splits 5 times and averaging ML evaluation metrics.
# LOW PRIORITY — dissertation validation support only.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ledeta.case_builder import build_cases
from ledeta.rubric import score_case_rubric_v3
from ledeta.model import train_or_load_model, predict_cases_with_rubric_comparison

RANDOM_STATE = 42
N_SPLITS = 5
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "enron_df_clean_full.csv")

def load_and_score_cases(path):
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cases = build_cases(df, window_days=30)
    scored = []
    for c in cases:
        res = score_case_rubric_v3(c)
        c_out = dict(c)
        c_out["rubric_score"] = float(res.get("triage_score", 0.0))
        scored.append(c_out)
    return scored

def band_label(score):
    if score <= 25: return "Low"
    elif score <= 50: return "Medium"
    elif score <= 75: return "High"
    return "Critical"

def run():
    print(f"Loading and scoring cases from: {DATASET_PATH}")
    cases = load_and_score_cases(DATASET_PATH)
    print(f"Total cases: {len(cases)}")

    scores = np.array([c["rubric_score"] for c in cases])
    bands = np.array([band_label(s) for s in scores])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(cases, bands), 1):
        train_cases = [cases[i] for i in train_idx]
        test_cases  = [cases[i] for i in test_idx]

        print(f"\n--- Fold {fold_idx} | Train: {len(train_cases)} | Test: {len(test_cases)} ---")

        train_res = train_or_load_model(train_cases, force_retrain=True)
        df_eval, metrics = predict_cases_with_rubric_comparison(train_res, test_cases, sample_n=None)

        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  N:    {metrics['n_eval']}")
        fold_metrics.append(metrics)

    maes  = [m["mae"]  for m in fold_metrics]
    rmses = [m["rmse"] for m in fold_metrics]
    r2s   = [m["r2"]   for m in fold_metrics]

    print("\n========== 5-Fold Average Results ==========")
    print(f"  Avg MAE:  {np.mean(maes):.4f}  (±{np.std(maes):.4f})")
    print(f"  Avg RMSE: {np.mean(rmses):.4f}  (±{np.std(rmses):.4f})")
    print(f"  Avg R²:   {np.mean(r2s):.4f}  (±{np.std(r2s):.4f})")
    print("==============================================")
    
    # Save 5-fold average results to CSV
    results_df = pd.DataFrame([
        {"metric": "MAE",  "fold_1": maes[0],  "fold_2": maes[1],  "fold_3": maes[2],  "fold_4": maes[3],  "fold_5": maes[4],  "avg": np.mean(maes),  "std": np.std(maes)},
        {"metric": "RMSE", "fold_1": rmses[0], "fold_2": rmses[1], "fold_3": rmses[2], "fold_4": rmses[3], "fold_5": rmses[4], "avg": np.mean(rmses), "std": np.std(rmses)},
        {"metric": "R2",   "fold_1": r2s[0],   "fold_2": r2s[1],   "fold_3": r2s[2],   "fold_4": r2s[3],   "fold_5": r2s[4],   "avg": np.mean(r2s),   "std": np.std(r2s)},
    ])

    out_path = os.path.join(os.path.dirname(__file__), "..", "kfold_validation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    run()