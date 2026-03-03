# tests/run_rq1_predictive_validity.py
import argparse
import numpy as np
import pandas as pd

GLOBAL_RANDOM_STATE = 42

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ledeta.case_builder import build_cases
from ledeta.rubric import score_case_rubric
from ledeta.features import extract_engineered_features, DEFAULT_KEYWORDS

# Stats
from scipy.stats import shapiro, ttest_rel, wilcoxon


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def paired_cohens_d(diff):
    # Cohen's d for paired samples = mean(diff)/sd(diff)
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")

def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)

def cases_to_Xy(cases):
    # Feature frame
    rows = []
    y = []
    for c in cases:
        feats = extract_engineered_features(c, keywords=DEFAULT_KEYWORDS)
        rows.append({k: float(v) if v is not None else np.nan for k, v in feats.items()})
        s, _ = score_case_rubric(c)   # rubric triage score (0..100)
        y.append(float(s))
    X = pd.DataFrame(rows)
    y = np.array(y, dtype=float)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to cleaned Enron CSV")
    ap.add_argument("--window_days", type=int, default=30)
    ap.add_argument("--body_mode", default="excerpt", choices=["excerpt", "full", "none"])
    ap.add_argument("--excerpt_len", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    cases = build_cases(df, window_days=args.window_days, body_mode=args.body_mode, excerpt_len=args.excerpt_len)
    X, y = cases_to_Xy(cases)

    # LightGBM model
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=args.seed,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    pipe.fit(X_train, y_train)

    y_pred = np.clip(pipe.predict(X_val), 0, 100)

    # Metrics
    mae_v = float(mean_absolute_error(y_val, y_pred))
    rmse_v = rmse(y_val, y_pred)
    r2_v = float(r2_score(y_val, y_pred))

    # Paired differences test
    diff = y_pred - y_val
    sh_w, sh_p = shapiro(diff)

    if sh_p >= 0.05:
        test_name = "paired_t"
        stat, p = ttest_rel(y_pred, y_val)
    else:
        test_name = "wilcoxon"
        stat, p = wilcoxon(y_pred, y_val, zero_method="wilcox", correction=False)

    d = paired_cohens_d(diff)

    # Bootstrap CIs
    mae_ci = bootstrap_ci(lambda a,b: float(mean_absolute_error(a,b)), y_val, y_pred, n_boot=args.boot, seed=args.seed)
    rmse_ci = bootstrap_ci(lambda a,b: rmse(a,b), y_val, y_pred, n_boot=args.boot, seed=args.seed)
    r2_ci = bootstrap_ci(lambda a,b: float(r2_score(a,b)), y_val, y_pred, n_boot=args.boot, seed=args.seed)

    out = {
        "n_total_cases": int(len(y)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "mae": mae_v, "mae_ci_lo": mae_ci[0], "mae_ci_hi": mae_ci[1],
        "rmse": rmse_v, "rmse_ci_lo": rmse_ci[0], "rmse_ci_hi": rmse_ci[1],
        "r2": r2_v, "r2_ci_lo": r2_ci[0], "r2_ci_hi": r2_ci[1],
        "shapiro_w": float(sh_w), "shapiro_p": float(sh_p),
        "paired_test": test_name, "test_stat": float(stat), "test_p": float(p),
        "cohens_d_paired": d,
    }

    print("\nRQ1 RESULTS (Predictive Validity)")
    for k,v in out.items():
        print(f"{k}: {v}")

    pd.DataFrame([out]).to_csv("rq1_results.csv", index=False)
    np.savetxt("rq1_y_val.csv", y_val, delimiter=",")
    np.savetxt("rq1_y_pred.csv", y_pred, delimiter=",")

    print("\nSaved: rq1_results.csv, rq1_y_val.csv, rq1_y_pred.csv")

if __name__ == "__main__":
    main()