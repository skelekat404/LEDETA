# ledeta/model.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ledeta.rubric import score_case_rubric
from ledeta.features import extract_engineered_features, DEFAULT_KEYWORDS


DEFAULT_MODEL_DIR = "ledeta_models"
DEFAULT_MODEL_NAME = "rubric_regressor.joblib"


def _try_import_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None


LGBM = _try_import_lgbm()


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, float]
    feature_names: List[str]
    model_path: str
    model_kind: str = "lightgbm"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cases_to_feature_frame(cases: List[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for c in cases:
        feats = extract_engineered_features(c, keywords=DEFAULT_KEYWORDS)
        row: dict = {}
        for k, v in feats.items():
            if v is None:
                row[k] = np.nan
            elif isinstance(v, (int, float, np.number)):
                row[k] = float(v)
            else:
                try:
                    row[k] = float(v)
                except Exception:
                    row[k] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_lgbm(random_state: int):
    """
    Build a LightGBM regressor with sane defaults for tabular regression.

    Note: Tree models do not need scaling. We keep an imputer.
    """
    if LGBM is None:
        raise ImportError(
            "LightGBM is not available in this environment. "
            "Install it with: pip install lightgbm"
        )

    model = LGBM.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        deterministic=True,
        n_jobs=-1,
    )
    return model


def _rmse(y_true, y_pred) -> float:
    # Avoid sklearn's squared= kw (older versions error). Use sqrt(MSE).
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_or_load_model(
    cases: List[dict],
    model_dir: str = DEFAULT_MODEL_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    force_retrain: bool = False,
    random_state: int = 42,
) -> TrainResult:
    """
    Train a LightGBM regressor to predict the rubric triage score,
    or load the saved one.

    Safety:
      - Retrain if feature columns changed
      - Retrain if joblib load fails (sklearn/lightgbm version mismatch)
    """
    _ensure_dir(model_dir)
    model_path = os.path.join(model_dir, model_name)

    X = _cases_to_feature_frame(cases)
    feature_names = list(X.columns)

    # y from rubric (triage score)
    y: List[float] = []
    for c in cases:
        s, _ = score_case_rubric(c)
        y.append(float(s))

    should_retrain = bool(force_retrain)

    # Load if possible
    if (not should_retrain) and os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            saved_features = bundle.get("feature_names", [])
            saved_kind = bundle.get("model_kind", "unknown")

            # Must match BOTH features and model kind, otherwise retrain
            if saved_features and saved_features == feature_names and saved_kind == "lightgbm":
                return TrainResult(
                    model=bundle["model"],
                    metrics=bundle.get("metrics", {}),
                    feature_names=saved_features,
                    model_path=model_path,
                    model_kind="lightgbm",
                )
            should_retrain = True
        except Exception as e:
            # Common: pickle incompat (sklearn internal symbol errors), or lightgbm mismatch
            print(f"[LEDETA] Could not load saved model ({e}). Retraining...")
            should_retrain = True

    # Train
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    lgbm_model = _build_lgbm(random_state=random_state)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgbm_model),
        ]
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    preds = np.clip(preds, 0, 100)

    mae = float(mean_absolute_error(y_val, preds))
    rmse = _rmse(y_val, preds)
    r2 = float(r2_score(y_val, preds))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "retrained": bool(should_retrain),
        "model_kind": "lightgbm",
    }

    joblib.dump(
        {
            "model": pipe,
            "metrics": metrics,
            "feature_names": feature_names,
            "model_kind": "lightgbm",
        },
        model_path,
    )

    return TrainResult(
        model=pipe,
        metrics=metrics,
        feature_names=feature_names,
        model_path=model_path,
        model_kind="lightgbm",
    )


def predict_cases(
    train_result_or_model: Any,
    cases: List[dict],
) -> pd.DataFrame:
    if hasattr(train_result_or_model, "model"):
        model = train_result_or_model.model
        feature_names = train_result_or_model.feature_names
    else:
        model = train_result_or_model
        feature_names = None

    X = _cases_to_feature_frame(cases)

    if feature_names:
        # Ensure column alignment for safety
        for col in feature_names:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feature_names]

    yhat = model.predict(X)
    yhat = np.clip(yhat, 0, 100)

    out = []
    for c, score in zip(cases, yhat):
        c_out = dict(c)
        c_out["priority_score"] = float(score)
        c_out["priority_band"] = pd.cut(
            [score],
            bins=[-1, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Critical"],
        )[0]
        out.append(c_out)

    return pd.DataFrame(out)


def predict_cases_with_rubric_comparison(
    train_result: TrainResult,
    cases: List[dict],
    sample_n: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    df_pred = predict_cases(train_result, cases).rename(
        columns={"priority_score": "ml_score", "priority_band": "ml_band"}
    )

    df_pred["rubric_score"] = np.nan
    df_pred["rubric_band"] = pd.NA
    df_pred["abs_error"] = np.nan

    if sample_n is None or sample_n <= 0 or sample_n >= len(df_pred):
        eval_idx = df_pred.index
    else:
        eval_idx = df_pred.sample(n=sample_n, random_state=42).index

    case_by_id = {c["case_id"]: c for c in cases}

    for i in eval_idx:
        cid = df_pred.at[i, "case_id"]
        c = case_by_id.get(cid)
        if c is None:
            continue

        s, _ = score_case_rubric(c)
        df_pred.at[i, "rubric_score"] = float(s)
        df_pred.at[i, "rubric_band"] = pd.cut(
            [s],
            bins=[-1, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Critical"],
        )[0]

    eval_mask = df_pred["rubric_score"].notna()
    df_pred.loc[eval_mask, "abs_error"] = (
        df_pred.loc[eval_mask, "ml_score"] - df_pred.loc[eval_mask, "rubric_score"]
    ).abs()

    eval_df = df_pred.loc[eval_mask].copy()
    y_true = eval_df["rubric_score"].astype(float).to_numpy()
    y_pred = eval_df["ml_score"].astype(float).to_numpy()

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(eval_df) else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(eval_df) else float("nan"),
        "r2": float(r2_score(y_true, y_pred)) if len(eval_df) else float("nan"),
        "n_eval": int(len(eval_df)),
        "model_kind": "lightgbm",
    }

    return df_pred, metrics
