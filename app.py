# app.py
import os
import json
import time
from datetime import datetime


import random
GLOBAL_RANDOM_STATE = 42

# Enforce deterministic behavior
os.environ["PYTHONHASHSEED"] = str(GLOBAL_RANDOM_STATE)
random.seed(GLOBAL_RANDOM_STATE)
np.random.seed(GLOBAL_RANDOM_STATE)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from ledeta.case_builder import build_cases
from ledeta.rubric import score_case_rubric_v3  # v3 dict output
from ledeta.model import train_or_load_model
from ledeta.model import predict_cases_with_rubric_comparison

from ledeta.explain import explain_case
from ledeta.audit import AuditLogger


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="LEDETA", layout="wide")

st.title("LEDETA — Law Enforcement Digital Evidence Triage Assistant")
st.caption("Case-level triage for text-based evidence (emails). Case = employee + 30-day window.")


# -----------------------------
# Helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _append_jsonl(path: str, payload: dict):
    """Append one JSONL record; never crash the app if this fails."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        st.warning(f"Could not write to audit log: {e}")


def _band_counts_chart(df: pd.DataFrame, mode: str):
    band_order = ["Low", "Medium", "High", "Critical"]

    if mode.startswith("Rubric"):
        counts = (
            df["display_band"]
            .astype(str)
            .value_counts()
            .reindex(band_order, fill_value=0)
            .reset_index()
        )
        counts.columns = ["band", "count"]

        return (
            alt.Chart(counts)
            .mark_bar(color="blue")
            .encode(
                x=alt.X("band:N", sort=band_order, title="Band"),
                y=alt.Y("count:Q", title="Cases"),
                tooltip=["band:N", "count:Q"],
            )
            .properties(height=220)
        )

    band_counts = pd.DataFrame({
        "band": band_order,
        "Rubric": df.get("rubric_band", pd.Series(dtype=str)).astype(str).value_counts().reindex(band_order, fill_value=0).values,
        "ML": df.get("ml_band", pd.Series(dtype=str)).astype(str).value_counts().reindex(band_order, fill_value=0).values,
    })

    band_long = band_counts.melt("band", var_name="source", value_name="count")

    return (
        alt.Chart(band_long)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=band_order, title="Band"),
            y=alt.Y("count:Q", title="Cases"),
            xOffset=alt.XOffset("source:N"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=["Rubric", "ML"], range=["blue", "red"]),
                legend=alt.Legend(title="")
            ),
            tooltip=["band:N", "source:N", "count:Q"],
        )
        .properties(height=220)
    )


def _ml_scatter_chart(df_eval: pd.DataFrame):
    d = df_eval.copy()
    d["rubric_score"] = pd.to_numeric(d["rubric_score"], errors="coerce")
    d["ml_score"] = pd.to_numeric(d["ml_score"], errors="coerce")
    d = d.dropna(subset=["rubric_score", "ml_score"])

    if d.empty:
        return None

    min_v = float(min(d["rubric_score"].min(), d["ml_score"].min()))
    max_v = float(max(d["rubric_score"].max(), d["ml_score"].max()))

    base = alt.Chart(d).properties(height=280)

    points = base.mark_circle(size=35, opacity=0.85, color="red").encode(
        x=alt.X("rubric_score:Q", title="Rubric score (ground truth)"),
        y=alt.Y("ml_score:Q", title="ML predicted score"),
        tooltip=[
            alt.Tooltip("case_id:N", title="Case"),
            alt.Tooltip("employee:N", title="Employee"),
            alt.Tooltip("rubric_score:Q", title="Rubric", format=".2f"),
            alt.Tooltip("ml_score:Q", title="ML", format=".2f"),
            alt.Tooltip("abs_error:Q", title="Abs error", format=".2f"),
        ],
    )

    diag = alt.Chart(pd.DataFrame({"x": [min_v, max_v], "y": [min_v, max_v]})).mark_line(
        color="blue", strokeDash=[6, 4]
    ).encode(x="x:Q", y="y:Q")

    return diag + points


# -----------------------------
# Sidebar controls  ✅ KEEP EVERYTHING INSIDE THIS BLOCK
# -----------------------------
with st.sidebar:
    st.header("Data")

    data_source = st.radio(
        "Data source",
        ["Upload CSV", "Local file path"],
        index=0,
        key="data_source_radio",
    )

    uploaded = None
    local_path = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload emails CSV", type=["csv"])
    else:
        local_path = st.text_input(
            "Local CSV path",
            value="",
            placeholder=r"C:\Users\mikec\OneDrive\Desktop\Doc School\Dissertation Build\datasets\enron_df_clean_full.csv",
        )

    st.divider()
    st.header("Triage Mode")
    triage_mode = st.radio(
        "How should LEDETA prioritize?",
        ["Rubric only (proxy ground truth)", "ML model trained to predict rubric score"],
        index=0,
    )

    st.divider()
    st.header("Queue Ranking")
    if triage_mode.startswith("Rubric"):
        rank_mode = st.radio(
            "Rank cases by",
            ["Triage score (ethics minus spam)", "Ethics score (ignore spam penalty)"],
            index=0,
        )
    else:
        rank_mode = "Triage score (ethics minus spam)"  # ML predicts triage

    st.divider()
    st.header("Spam handling")
    include_spam_filtered = st.checkbox("Include spam-filtered cases", value=False)
    st.caption("Default filters out newsletter/promo-dominant cases with low ethics signal.")

    st.divider()
    st.header("Case Definition")
    window_days = st.number_input("Window size (days)", min_value=7, max_value=60, value=30, step=1)

    st.divider()
    st.header("ML Evaluation")
    if triage_mode.startswith("ML"):
        force_retrain = st.checkbox("Force retrain ML model", value=False)
        ml_eval_sample_n = st.number_input(
            "Rubric comparison sample size (0 = all cases)",
            min_value=0,
            value=500,
            step=50,
        )
    else:
        force_retrain = False
        ml_eval_sample_n = 0

    st.divider()
    st.header("Audit")
    run_id = st.text_input("Run ID (optional)", value="")
    log_run_metrics = st.checkbox("Log runtime metrics to audit log", value=True)

run_id = (run_id or "").strip()


# -----------------------------
# Load emails
# -----------------------------
@st.cache_data(show_spinner=False)
def load_emails(uploaded_file, local_path: str | None) -> pd.DataFrame:
    if uploaded_file is None and (not local_path):
        return pd.DataFrame(columns=["employee", "date", "subject", "body"])

    desired = [
        "employee", "date", "subject", "body",
        "folder", "body_length", "subject_length",
        "from", "to", "cc", "bcc", "message_id", "file"
    ]

    def _read_header(path_or_buf):
        return pd.read_csv(path_or_buf, nrows=0)

    if uploaded_file is not None:
        header = _read_header(uploaded_file)
        uploaded_file.seek(0)
        cols = [c for c in desired if c in header.columns]
        df = pd.read_csv(uploaded_file, usecols=cols, low_memory=False)
    else:
        if not os.path.exists(local_path):
            return pd.DataFrame(columns=["employee", "date", "subject", "body"])

        header = _read_header(local_path)
        cols = [c for c in desired if c in header.columns]

        chunks = []
        for chunk in pd.read_csv(local_path, usecols=cols, low_memory=False, chunksize=200_000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


t0 = time.perf_counter()
emails = load_emails(uploaded, local_path)
t_load = time.perf_counter() - t0

if emails.empty:
    if data_source == "Local file path" and local_path and (not os.path.exists(local_path)):
        st.error(f"Local path not found:\n\n`{local_path}`")
    else:
        st.info("Upload a CSV (small) or enter a local file path (large) to begin.")
    st.stop()

required_cols = {"employee", "date", "subject", "body"}
missing = required_cols - set(emails.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()


# -----------------------------
# Build cases (CACHED)
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_build_cases(emails_df: pd.DataFrame, window_days_i: int):
    return build_cases(emails_df, window_days=window_days_i)


t0 = time.perf_counter()
cases = cached_build_cases(emails, int(window_days))
t_cases = time.perf_counter() - t0

# Fast O(1) case lookup
case_by_id = {c["case_id"]: c for c in cases}

st.subheader("Cases")
st.write(f"Built **{len(cases)}** cases from **{emails['employee'].nunique()}** employees.")


# -----------------------------
# Score cases (Rubric OR ML)
# -----------------------------
logger = AuditLogger(run_id=run_id or None)

t0 = time.perf_counter()
ml_eval_metrics = None
train_res = None

# ✅ Safety: if you haven't added the sidebar checkbox yet, default to False
# (If you *did* add it in the sidebar, this does nothing because it's already defined.)
try:
    include_spam_filtered
except NameError:
    include_spam_filtered = False


if triage_mode.startswith("Rubric"):
    cases_scored = []
    spam_filtered_count = 0

    for c in cases:
        res = score_case_rubric_v3(c)

        # HARD SPAM FILTER: drop newsletters/promos unless user opts in
        spam_filtered = bool(res.get("spam_filtered", False))
        if spam_filtered:
            spam_filtered_count += 1
            if not include_spam_filtered:
                continue

        triage_score = float(res.get("triage_score", 0.0))
        ethics_score = float(res.get("fraud_score", 0.0))  # legacy key; now ethics_score
        spam_penalty = float(res.get("spam_penalty", 0.0))
        reasons = res.get("reasons", [])

        c_out = dict(c)
        c_out["triage_score"] = triage_score
        c_out["fraud_score"] = ethics_score        # legacy key, kept for compatibility
        c_out["ethics_score"] = ethics_score       # explicit alias (new)
        c_out["spam_penalty"] = spam_penalty
        c_out["spam_filtered"] = spam_filtered
        c_out["priority_score"] = triage_score     # backward compat

        c_out["triage_band"] = pd.cut(
            [triage_score], bins=[-1, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Critical"]
        )[0]
        c_out["fraud_band"] = pd.cut(
            [ethics_score], bins=[-1, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Critical"]
        )[0]

        c_out["priority_band"] = c_out["triage_band"]
        c_out["rubric_reasons"] = reasons
        cases_scored.append(c_out)

    df_cases = pd.DataFrame(cases_scored)

    # ✅ Ensure columns exist / are clean (helps export + filters)
    if not df_cases.empty:
        df_cases["spam_filtered"] = df_cases.get("spam_filtered", False).fillna(False)

    if include_spam_filtered:
        st.caption(f"Spam-filtered cases included: {spam_filtered_count}")
    else:
        st.caption(f"Spam-filtered cases removed from queue: {spam_filtered_count} (toggle in sidebar to include)")

    if df_cases.empty:
        st.warning("All cases were filtered out (likely spam/newsletters). Try enabling 'Include spam-filtered cases'.")
        st.stop()

    # ✅ Ranking logic:
    # Your sidebar option probably still says "Fraud score (ignore spam penalty)"
    # We map that choice to ethics_score now.
    if rank_mode.startswith("Fraud"):
        sort_col = "ethics_score"
        band_col = "fraud_band"
    else:
        sort_col = "triage_score"
        band_col = "triage_band"

    df_cases = df_cases.sort_values(sort_col, ascending=False)
    df_cases["display_score"] = df_cases[sort_col]
    # -----------------------------
    # Band calibration (CRITICAL FIX)
    # -----------------------------
    USE_QUANTILE_BANDS = True

    CRITICAL_PERCENTILE = 0.97   # top 3%
    HIGH_PERCENTILE = 0.85
    MED_PERCENTILE = 0.60
    CRITICAL_FLOOR = 85.0        # absolute minimum for Critical

    if USE_QUANTILE_BANDS and len(df_cases) >= 50:
        s = pd.to_numeric(df_cases["display_score"], errors="coerce").dropna()

        if len(s) >= 50:
            q_med = float(s.quantile(MED_PERCENTILE))
            q_high = float(s.quantile(HIGH_PERCENTILE))
            q_crit = float(s.quantile(CRITICAL_PERCENTILE))

            # enforce monotonic cutoffs
            cuts = np.maximum.accumulate([q_med, q_high, q_crit])
            q_med, q_high, q_crit = cuts.tolist()

            df_cases["display_band"] = pd.cut(
                df_cases["display_score"],
                bins=[-np.inf, q_med, q_high, q_crit, np.inf],
                labels=["Low", "Medium", "High", "Critical"],
                include_lowest=True,
            ).astype(str)

            # hard floor: demote weak Criticals
            crit_mask = (
                (df_cases["display_band"] == "Critical")
                & (df_cases["display_score"] < CRITICAL_FLOOR)
            )
            df_cases.loc[crit_mask, "display_band"] = "High"

            st.caption(
                f"Band calibration: Medium≥{q_med:.1f}, "
                f"High≥{q_high:.1f}, "
                f"Critical≥max({q_crit:.1f}, {CRITICAL_FLOOR:.1f})"
            )
    else:
        # fallback to original fixed bands
        df_cases["display_band"] = df_cases[band_col]


else:
    train_res = train_or_load_model(cases, force_retrain=bool(force_retrain))
    sample_n = None if int(ml_eval_sample_n) == 0 else int(ml_eval_sample_n)

    # --- Show what ML model is actually being used (LightGBM vs fallback) ---
    model_kind = None
    if hasattr(train_res, "metrics") and isinstance(train_res.metrics, dict):
        model_kind = train_res.metrics.get("model_kind", None)

    st.info(
        f"ML model loaded: **{model_kind or 'unknown'}** | "
        f"retrained: **{bool(train_res.metrics.get('retrained', False)) if hasattr(train_res, 'metrics') else 'unknown'}** | "
        f"path: `{getattr(train_res, 'model_path', '')}`"
    )

    df_eval, ml_eval_metrics = predict_cases_with_rubric_comparison(
        train_res,
        cases,
        sample_n=sample_n,
    )

    df_cases = df_eval.sort_values("ml_score", ascending=False)
    df_cases["display_score"] = df_cases["ml_score"]
    df_cases["display_band"] = df_cases["ml_band"]

t_score = time.perf_counter() - t0


# -----------------------------
# Investigator filters + export
# -----------------------------
t_filter_start = time.perf_counter()

st.sidebar.divider()
st.sidebar.header("Investigator Filters")

# NEW: toggle to hide spam/newsletter-dominant cases (rubric flag)
hide_spam_filtered = st.sidebar.checkbox(
    "Hide spam/newsletter-dominant cases",
    value=True,
    help="Uses rubric spam_filtered flag to remove obvious newsletters/promos by default."
)

df_cases["window_start"] = pd.to_datetime(df_cases["window_start"], errors="coerce")
df_cases["window_end"] = pd.to_datetime(df_cases["window_end"], errors="coerce")

all_bands = ["Low", "Medium", "High", "Critical"]
selected_bands = st.sidebar.multiselect("Priority bands", options=all_bands, default=all_bands)

employees_sorted = sorted(df_cases["employee"].dropna().unique().tolist())
selected_employees = st.sidebar.multiselect("Employees", options=employees_sorted, default=[])

min_date = df_cases["window_start"].min()
max_date = df_cases["window_start"].max()

if pd.isna(min_date) or pd.isna(max_date):
    date_range = None
else:
    date_range = st.sidebar.date_input(
        "Case window start date range",
        value=(min_date.date(), max_date.date()),
    )

df_filtered = df_cases.copy()

# Apply filters (bands, employees)
if selected_bands:
    df_filtered = df_filtered[df_filtered["display_band"].astype(str).isin(selected_bands)]
if selected_employees:
    df_filtered = df_filtered[df_filtered["employee"].isin(selected_employees)]

# Apply date range filter
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    start_d, end_d = date_range
    start_ts = pd.to_datetime(start_d)
    end_ts = pd.to_datetime(end_d) + pd.Timedelta(days=1)

    ws = df_filtered["window_start"]
    if getattr(ws.dt, "tz", None) is not None:
        tz = ws.dt.tz
        start_ts = start_ts.tz_localize(tz)
        end_ts = end_ts.tz_localize(tz)

    df_filtered = df_filtered[
        (df_filtered["window_start"] >= start_ts) &
        (df_filtered["window_start"] < end_ts)
    ]

# NEW: spam/newsletter filter AFTER all other filters
if hide_spam_filtered and "spam_filtered" in df_filtered.columns:
    df_filtered = df_filtered[~df_filtered["spam_filtered"].fillna(False)]

t_filter = time.perf_counter() - t_filter_start

if df_filtered.empty:
    st.warning("No cases match your filters. Try expanding bands, employee list, date range, or uncheck 'Hide spam'.")
    st.stop()

# Export
base_export_cols = ["case_id", "employee", "window_start", "window_end", "n_emails"]
if triage_mode.startswith("Rubric"):
    wanted_export = base_export_cols + ["triage_score", "triage_band", "ethics_score", "spam_penalty", "spam_filtered"]
else:
    wanted_export = base_export_cols + ["ml_score", "ml_band", "rubric_score", "rubric_band", "abs_error"]

export_cols = [c for c in wanted_export if c in df_filtered.columns]

csv_bytes = df_filtered[export_cols].to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download filtered queue (CSV)",
    data=csv_bytes,
    file_name="ledeta_ranked_queue_filtered.csv",
    mime="text/csv",
)



# -----------------------------
# Runtime metrics (top)
# -----------------------------
n_emails_total = int(len(emails))
n_cases_total = int(len(cases))
cases_per_sec = (n_cases_total / t_cases) if t_cases > 0 else 0.0
emails_per_sec = (n_emails_total / t_load) if t_load > 0 else 0.0

score_label = "Score/model time (s)" if triage_mode.startswith("Rubric") else "ML train/predict time (s)"

with st.expander("⏱️ Runtime metrics", expanded=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Load time (s)", f"{t_load:.2f}")
    m2.metric("Case build time (s)", f"{t_cases:.2f}")
    m3.metric(score_label, f"{t_score:.2f}")
    m4.metric("Filter time (s)", f"{t_filter:.2f}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Emails processed", f"{n_emails_total:,}")
    s2.metric("Cases built", f"{n_cases_total:,}")
    s3.metric("Cases/sec (build)", f"{cases_per_sec:.2f}")
    
    if triage_mode.startswith("ML") and train_res is not None:
        mk = None
        if hasattr(train_res, "metrics") and isinstance(train_res.metrics, dict):
            mk = train_res.metrics.get("model_kind", None)
        st.caption(
            f"ML model kind: {mk or 'unknown'} | "
            f"retrained: {train_res.metrics.get('retrained', False) if hasattr(train_res, 'metrics') else 'unknown'}"
        )
    

    st.caption(f"Emails/sec (load stage): {emails_per_sec:.2f}")


# -----------------------------
# Audit logging: run metrics
# -----------------------------
if log_run_metrics:
    run_payload = {
        "event": "run_metrics",
        "ts": _now_iso(),
        "run_id": run_id or None,
        "data_source": data_source,
        "window_days": int(window_days),
        "triage_mode": triage_mode,
        "n_emails": n_emails_total,
        "n_cases": n_cases_total,
        "t_load_s": float(t_load),
        "t_case_build_s": float(t_cases),
        "t_score_s": float(t_score),
        "t_filter_s": float(t_filter),
        "random_state": GLOBAL_RANDOM_STATE,
        "include_spam_filtered": bool(include_spam_filtered),
    }
    if ml_eval_metrics is not None:
        run_payload["ml_eval_metrics"] = ml_eval_metrics
        if train_res is not None:
            run_payload["ml_model_path"] = getattr(train_res, "model_path", None)

    _append_jsonl(logger.path, run_payload)


# -----------------------------
# Display table + summary chart (filtered)
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    if triage_mode.startswith("Rubric"):
        wanted_display = [
            "case_id", "employee", "window_start", "window_end", "n_emails",
            "display_score", "display_band",
            "triage_score", "triage_band",
            "ethics_score",
            "spam_penalty", "spam_filtered",
        ]
    else:
        wanted_display = ["case_id", "employee", "window_start", "window_end", "n_emails", "ml_score", "ml_band", "rubric_score", "abs_error"]

    display_cols = [c for c in wanted_display if c in df_filtered.columns]
    st.dataframe(df_filtered[display_cols], use_container_width=True, hide_index=True)

with col2:
    st.metric("Top score", float(df_filtered["display_score"].max()))
    st.metric("Median score", float(df_filtered["display_score"].median()))
    st.altair_chart(_band_counts_chart(df_filtered, triage_mode), use_container_width=True)
    if triage_mode.startswith("ML"):
        st.caption("Color key: **Rubric = blue**, **ML = red**")


# -----------------------------
# ML vs Rubric evaluation panel (ML mode only)
# -----------------------------
if (not triage_mode.startswith("Rubric")) and (ml_eval_metrics is not None):
    st.divider()
    st.subheader("ML vs Rubric (ground truth) — Evaluation")

    a, b, c, d = st.columns(4)
    a.metric("MAE", f"{ml_eval_metrics['mae']:.2f}")
    b.metric("RMSE", f"{ml_eval_metrics['rmse']:.2f}")
    c.metric("R²", f"{ml_eval_metrics['r2']:.3f}")
    d.metric("N evaluated", f"{ml_eval_metrics['n_eval']:,}")

    chart_df = df_cases[["case_id", "employee", "rubric_score", "ml_score", "abs_error"]].dropna()
    sc = _ml_scatter_chart(chart_df)
    if sc is not None:
        st.altair_chart(sc, use_container_width=True)
        st.caption("Color key: **ML points = red**. Diagonal reference line = **rubric agreement (blue)**.")

    st.markdown("**Largest ML errors (top 10)**")
    if "abs_error" in df_cases.columns:
        st.dataframe(
            df_cases.sort_values("abs_error", ascending=False)[
                [c for c in ["case_id", "employee", "rubric_score", "ml_score", "abs_error", "window_start", "window_end", "n_emails"] if c in df_cases.columns]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------
# Case drill-down
# -----------------------------
st.divider()
st.subheader("Case drill-down")

selected = st.selectbox("Select a case", df_filtered["case_id"].tolist())
case_row = df_filtered[df_filtered["case_id"] == selected].iloc[0].to_dict()

if triage_mode.startswith("Rubric"):
    st.markdown(
        f"**Ethics score:** {float(case_row.get('ethics_score', case_row.get('fraud_score', 0.0))):.2f} | "
        f"**Triage score:** {float(case_row.get('triage_score', 0.0)):.2f} | "
        f"**Spam penalty:** {float(case_row.get('spam_penalty', 0.0)):.2f} | "
        f"**Spam filtered:** {bool(case_row.get('spam_filtered', False))}"
    )

# O(1) lookup
case_obj = case_by_id[selected]

st.markdown("### Emails in this case")

@st.cache_data(show_spinner=False)
def cached_case_email_headers(case_id: str, case_obj_dict: dict) -> pd.DataFrame:
    rows = []
    for idx, e in enumerate(case_obj_dict.get("emails", [])):
        rows.append({
            "_idx": idx,
            "date": e.get("date"),
            "folder": e.get("folder", ""),
            "from": e.get("from", ""),
            "to": e.get("to", ""),
            "subject": e.get("subject", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False)
    return df


case_emails = case_obj.get("emails", [])
if not case_emails:
    st.info("No emails were attached to this case object. (Check build_cases: it should store case['emails'].)")
else:
    q = st.text_input("Search within case emails (subject only — fast)", value="")

    df_case_emails = cached_case_email_headers(selected, case_obj)

    if q.strip() and (not df_case_emails.empty):
        qq = q.strip().lower()
        df_case_emails = df_case_emails[
            df_case_emails["subject"].fillna("").str.lower().str.contains(qq)
        ]

    st.dataframe(
        df_case_emails[["date", "folder", "from", "to", "subject"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### View full email")
    if len(df_case_emails):
        def _safe_str(x):
            return "" if x is None else str(x)

        email_idx = st.selectbox(
            "Select email",
            options=list(range(len(df_case_emails))),
            format_func=lambda i: f"{df_case_emails.iloc[i]['date']} — {_safe_str(df_case_emails.iloc[i]['subject'])[:80]}",
        )

        header_row = df_case_emails.iloc[email_idx].to_dict()
        raw_email = case_emails[int(header_row["_idx"])]

        st.markdown(f"**Date:** {header_row.get('date')}")
        st.markdown(f"**Folder:** {header_row.get('folder')}")
        st.markdown(f"**From:** {header_row.get('from')}")
        st.markdown(f"**To:** {header_row.get('to')}")
        st.markdown(f"**Subject:** {header_row.get('subject')}")
        st.text_area("Body", value=str(raw_email.get("body", "") or ""), height=260)

# -----------------------------
# Explanation + audit (button-controlled + cached)
# -----------------------------
case_row_for_explain = dict(case_row)
if not triage_mode.startswith("Rubric"):
    case_row_for_explain["priority_score"] = float(case_row_for_explain.get("ml_score", 0.0))
    case_row_for_explain["priority_band"] = case_row_for_explain.get("ml_band", "Unknown")

@st.cache_data(show_spinner=False)
def cached_explain(case_id: str, case_obj_dict: dict, row_for_explain: dict):
    return explain_case(case_obj_dict, row_for_explain)

st.divider()
st.subheader("Explanation")

do_explain = st.button("Generate explanation for this case", type="primary")

if do_explain:
    t0 = time.perf_counter()
    with st.spinner("Generating explanation..."):
        explanation = cached_explain(selected, case_obj, case_row_for_explain)
    t_explain = time.perf_counter() - t0

    logger.log_case(case_obj, case_row_for_explain, explanation)

    if log_run_metrics:
        _append_jsonl(logger.path, {
            "event": "case_explain_metrics",
            "ts": _now_iso(),
            "run_id": run_id or None,
            "case_id": selected,
            "t_explain_s": float(t_explain),
        })

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Summary")
        st.write(explanation.get("summary", ""))
        st.markdown("**Top drivers**")
        st.dataframe(pd.DataFrame(explanation.get("top_features", [])), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Evidence snippets")
        for snip in (explanation.get("snippets", [])[:6]):
            title = f"{snip.get('date', '')} — {snip.get('subject', '')}"
            if snip.get("folder"):
                title += f"  [{snip['folder']}]"
            with st.expander(title, expanded=False):
                st.write(snip.get("excerpt", ""))

else:
    st.info("Explanation generation is expensive. Click **Generate explanation** when you’re ready.")

st.markdown("### Audit log")
st.code(logger.path, language="text")
st.caption("Each run writes JSONL entries with timestamp, score, features, and explanation payload.")
