# tests/run_rq2_efficiency.py
import argparse
import time
import numpy as np
import pandas as pd

import os
import random

GLOBAL_RANDOM_STATE = 42

# Enforce deterministic behavior
os.environ["PYTHONHASHSEED"] = str(GLOBAL_RANDOM_STATE)
random.seed(GLOBAL_RANDOM_STATE)
np.random.seed(GLOBAL_RANDOM_STATE)

from scipy.stats import shapiro, ttest_rel, wilcoxon

from ledeta.case_builder import build_cases
from ledeta.rubric import score_case_rubric_v3
from ledeta.model import train_or_load_model, predict_cases_with_rubric_comparison


def rank_biserial_from_wilcoxon(x, y):
    # rank-biserial correlation for paired samples via Wilcoxon signed ranks
    # r_rb = (W_pos - W_neg) / (W_pos + W_neg)
    d = np.asarray(y) - np.asarray(x)
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))

def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, (m - i) * pvals[idx])
    # enforce monotonicity
    # sorted adjusted should be non-decreasing in the order of sorted pvals
    adj_sorted = adj[order]
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i-1])
    adj[order] = adj_sorted
    return adj.tolist()

def run_rubric(df_emails, window_days):
    t0 = time.perf_counter()
    cases = build_cases(df_emails, window_days=window_days)
    # score all cases
    scored = []
    for c in cases:
        res = score_case_rubric_v3(c)
        scored.append(res["triage_score"])
    t = time.perf_counter() - t0
    return t, len(df_emails), len(cases)

def run_ml(df_emails, window_days, force_retrain=False):
    t0 = time.perf_counter()
    cases = build_cases(df_emails, window_days=window_days)
    train_res = train_or_load_model(cases, force_retrain=force_retrain)
    # prediction only (evaluation sample off; we just need triage runtime)
    _df_pred, _metrics = predict_cases_with_rubric_comparison(train_res, cases, sample_n=0)
    t = time.perf_counter() - t0
    return t, len(df_emails), len(cases)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--window_days", type=int, default=30)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--force_retrain_each_run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["employee", "date"])

    rows = []
    for i in range(1, args.runs + 1):
        t_r, n_emails, n_cases = run_rubric(df, args.window_days)
        t_m, _, _ = run_ml(df, args.window_days, force_retrain=args.force_retrain_each_run)

        rows.append({
            "run": i,
            "rubric_total_s": t_r,
            "ml_total_s": t_m,
            "n_emails": n_emails,
            "n_cases": n_cases,
            "rubric_cases_per_s": n_cases / t_r,
            "ml_cases_per_s": n_cases / t_m,
            "rubric_emails_per_s": n_emails / t_r,
            "ml_emails_per_s": n_emails / t_m,
        })
        print(f"Run {i}/{args.runs}: rubric={t_r:.3f}s | ml={t_m:.3f}s")

    df_out = pd.DataFrame(rows)
    df_out.to_csv("rq2_runs.csv", index=False)

    # Stats (paired diffs: ML - Rubric) — test improvement
    results = []
    for dv_pair in [
        ("total_runtime_s", "rubric_total_s", "ml_total_s", "lower_is_better"),
        ("cases_per_s", "rubric_cases_per_s", "ml_cases_per_s", "higher_is_better"),
        ("emails_per_s", "rubric_emails_per_s", "ml_emails_per_s", "higher_is_better"),
    ]:
        name, col_r, col_m, direction = dv_pair
        r = df_out[col_r].to_numpy()
        m = df_out[col_m].to_numpy()
        diff = m - r

        sh_w, sh_p = shapiro(diff)

        if sh_p >= 0.05:
            test = "paired_t"
            stat, p = ttest_rel(m, r)
            eff = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")  # paired d
            eff_name = "cohens_d_paired"
        else:
            test = "wilcoxon"
            stat, p = wilcoxon(m, r, zero_method="wilcox", correction=False)
            eff = rank_biserial_from_wilcoxon(r, m)
            eff_name = "rank_biserial"

        results.append({
            "dv": name,
            "direction": direction,
            "shapiro_w": float(sh_w),
            "shapiro_p": float(sh_p),
            "test": test,
            "stat": float(stat),
            "p": float(p),
            eff_name: float(eff),
            "rubric_mean": float(r.mean()),
            "ml_mean": float(m.mean()),
            "mean_diff_ml_minus_rubric": float(diff.mean()),
        })

    # Holm correction across the 3 DVs
    pvals = [x["p"] for x in results]
    adj = holm_adjust(pvals)
    for i in range(len(results)):
        results[i]["p_holm"] = adj[i]

    df_res = pd.DataFrame(results)
    df_res.to_csv("rq2_stats.csv", index=False)

    print("\nRQ2 STATS")
    print(df_res.to_string(index=False))
    print("\nSaved: rq2_runs.csv, rq2_stats.csv")

if __name__ == "__main__":
    main()