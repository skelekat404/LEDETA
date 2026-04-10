# app.py
# DESCRIPTION: app.py operationalizes the entire LEDETA workflow. It loads structured email data, builds case-level units, 
# applies either rubric-only or ML-assisted prioritization, allows the user to filter and inspect the queue, generates 
# case-level explanations, and writes audit records. Methodologically, it supports my Chapter 3 system design, human-in-the-loop 
# framework, reproducibility controls, and governance requirements. Empirically, it supports Chapter 4 by surfacing runtime 
# metrics, predictive validity metrics, and practical usability features.

import os #Python's OS module, work with file paths, env variables and dirs
import json #audit records can be written as JSON text
import time #measure runtime for loading, case buildingg, scoring, etc
from datetime import datetime #timestamps for audit logging

import numpy as np #numerical operations 
import random #can control random behavior with fixed seed

# Set a fixed random state to improve reproducibility across runs. 
# This supports the dissertation's emphasis on controlled evaluation, traceability, and consistency.
GLOBAL_RANDOM_STATE = 42 # fixed seed

# Enforce deterministic behavior
os.environ["PYTHONHASHSEED"] = str(GLOBAL_RANDOM_STATE) #set Python's hash seed so hasing behavior is deterministic across runs
random.seed(GLOBAL_RANDOM_STATE) #set seed for Python's built in random number generator
np.random.seed(GLOBAL_RANDOM_STATE) #set numpy's random seed to it is also repeatable

import streamlit as st #framework for building interactive web app
import pandas as pd #table/dataframe manipulation
import altair as alt #charts in the interface

from ledeta.case_builder import build_cases #imports build_cases function to transform raw email rows into case-level units
from ledeta.rubric import score_case_rubric_v3  # imports rubric scoring function, for case-level proxy ground-truth score
from ledeta.model import train_or_load_model #function to load or retrain a saved ML model
from ledeta.model import predict_cases_with_rubric_comparison #function that generates ML predictions and compares against rubric scores

from ledeta.explain import explain_case #function that creates case explanation output
from ledeta.audit import AuditLogger #Audit logger class used to write the structured logs


# -----------------------------
# Page config
# -----------------------------

# Streamlit page setup.
# The caption explicitly states the dissertation's unit of analysis:
# a case is one employee's emails within a 30-day window.
st.set_page_config(page_title="LEDETA", layout="wide") #sets Streamlit browser tab title and wide page layout so app has more screen space

st.title("LEDETA — Law Enforcement Digital Evidence Triage Assistant") #displays main title at top of app
st.caption("Case-level triage for text-based evidence (emails). Case = employee + 30-day window.") #short explanatory caption under title


# -----------------------------
# Helpers
# -----------------------------

# Return an ISO timestamp for audit records and runtime logging.
def _now_iso() -> str: #helper funtion: returns a timestamp string
    return datetime.utcnow().isoformat() + "Z" #gets current UTC time, converts to ISO, appends to Z to indicate UTC. used in audit logs

# Append structured JSONL audit records without crashing the app.
# This supports auditability and governance in a defensible workflow.
def _append_jsonl(path: str, payload: dict): #helper function: write one JSON record to a file. path: file loc, payload: dict to log
    """Append one JSONL record; never crash the app if this fails."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) #creates dir for log file (if not exist). exist_ok=true prevents error if folder exists
        with open(path, "a", encoding="utf-8") as f: #opens file in append mode so new records are added at end using UTF-8 encoding
            f.write(json.dumps(payload, ensure_ascii=False) + "\n") #converts payload dict into JSON string and writes as one line
    except Exception as e: #if anything goes wrong, catch exception
        st.warning(f"Could not write to audit log: {e}") #warning message instead of crashing

# Visualize the distribution of severity levels so users can quickly
# understand queue composition under rubric-only or ML-assisted modes.
def _band_counts_chart(df: pd.DataFrame, mode: str): #function to build bar chart showing how many cases fall into each severity level.
    band_order = ["Low", "Medium", "High", "Critical"] #defines desired order of the bands so charts always display in sequence

    if mode.startswith("Rubric"): #check whether app is in rubric mode
        counts = ( #builds a table of counts for each display band
            df["display_band"]
            .astype(str) #ensures values are strings
            .value_counts() #counts each band
            .reindex(band_order, fill_value=0) #forces the output into low->critical order and fills missing bands with 0
            .reset_index() #turns result into regular df
        )
        counts.columns = ["severity", "count"]

        bars = (
            alt.Chart(counts)
            .mark_bar(color="blue")
            .encode(
                x=alt.X("severity:N", sort=band_order, title="Severity"),
                y=alt.Y(
                    "count:Q",
                    title="Cases",
                    scale=alt.Scale(domain=[0, counts["count"].max() * 1.12])
                ),
                tooltip=["severity:N", "count:Q"],
            )
        )

        labels = (
            alt.Chart(counts)
            .mark_text(dy=-6, color="white")
            .encode(
                x=alt.X("severity:N", sort=band_order),
                y=alt.Y(
                    "count:Q",
                    scale=alt.Scale(domain=[0, counts["count"].max() * 1.12])
                ),
                text=alt.Text("count:Q", format=".0f"),
            )
        )

        return (bars + labels).properties(height=220)

    band_counts = pd.DataFrame({ #if not in rubric mode, this builds a df with 1 row per band and 2 count columns
        "severity": band_order,
        "Rubric": df.get("rubric_band", pd.Series(dtype=str)).astype(str).value_counts().reindex(band_order, fill_value=0).values, #handles missing columns
        "ML": df.get("ml_band", pd.Series(dtype=str)).astype(str).value_counts().reindex(band_order, fill_value=0).values, #handles missing columns
    })

    band_long = band_counts.melt("severity", var_name="source", value_name="count") #converts wide df into long format so Altair can plot grouped bars by source (rubric vs ML)

    bars = (
        alt.Chart(band_long)
        .mark_bar()
        .encode(
            x=alt.X("severity:N", sort=band_order, title="Severity"),
            y=alt.Y(
                "count:Q",
                title="Cases",
                scale=alt.Scale(domain=[0, band_long["count"].max() * 1.12])
            ),
            xOffset=alt.XOffset("source:N"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=["Rubric", "ML"], range=["blue", "red"]),
                legend=alt.Legend(title="")
            ),
            tooltip=["severity:N", "source:N", "count:Q"],
        )
    )

    labels = (
        alt.Chart(band_long)
        .mark_text(dy=-8, color="white")
        .encode(
            x=alt.X("severity:N", sort=band_order),
            y=alt.Y(
                "count:Q",
                scale=alt.Scale(domain=[0, band_long["count"].max() * 1.12])
            ),
            xOffset=alt.XOffset("source:N"),
            text=alt.Text("count:Q", format=".0f"),
            detail="source:N",
        )
    )

    return (bars + labels).properties(height=220)

# Plot ML-predicted scores against rubric-derived scores to show
# agreement between the model and the dissertation's proxy ground truth.
def _ml_scatter_chart(df_eval: pd.DataFrame): #function to build ML-vs-rubric scatterplot
    d = df_eval.copy()

    # Add very small jitter to reduce overplotting (does not change underlying values)
    rng = np.random.default_rng(GLOBAL_RANDOM_STATE)
    d["rubric_score_jitter"] = d["rubric_score"] + rng.uniform(-0.3, 0.3, size=len(d))
    d["ml_score_jitter"] = d["ml_score"] + rng.uniform(-0.3, 0.3, size=len(d))
    d["rubric_score"] = pd.to_numeric(d["rubric_score"], errors="coerce") #converts rubric_score column to numeric values, invalid entries become NaN
    d["ml_score"] = pd.to_numeric(d["ml_score"], errors="coerce") #same conversion
    d = d.dropna(subset=["rubric_score", "ml_score"]) #drops rows where either score is missing, because those rows cannot be plotted meaningfully

    if d.empty: #no valid rows left, return None instead of trying to plot empty chart
        return None

    min_v = float(min(d["rubric_score"].min(), d["ml_score"].min())) #finds smallest value across both axes so diagonal reference line starts at the right minimum
    max_v = float(max(d["rubric_score"].max(), d["ml_score"].max())) #finds largest value across both acrs so the diagonal line ends at right max

    base = alt.Chart(d).properties(height=280).encode(
        x=alt.X(scale=alt.Scale(domain=[0, 100]))
    ) #creates base Altair chart obj using cleaned df and sets chart height

    points = base.mark_circle(size=35, opacity=0.65, color="red").encode(
        x=alt.X("rubric_score_jitter:Q", title="Rubric score (ground truth)"),
        y=alt.Y("ml_score_jitter:Q", title="ML predicted score"),
        tooltip=[ #defines what will appear when each part is hovered over
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

    cutoff_df = pd.DataFrame({
        "x": [25, 50, 75],
        "label": ["Low/Medium", "Medium/High", "High/Critical"],
    })

    cutoffs = (
        alt.Chart(cutoff_df)
        .mark_rule(color="yellow", strokeWidth=3)
        .encode(
            x=alt.X("x:Q"),
            tooltip=["label:N", "x:Q"],
        )
    )
    # Region label annotations — placed at midpoints of each score band
    region_labels_df = pd.DataFrame({
        "x": [12.5, 37.5, 62.5, 87.5],
        "y": [3.0, 3.0, 3.0, 3.0],
        "label": ["Low", "Medium", "High", "Critical"],
    })
    region_labels = (
        alt.Chart(region_labels_df)
        .mark_text(fontSize=11, fontWeight="bold", color="yellow", opacity=0.85)
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            text=alt.Text("label:N"),
        )
    )

    return cutoffs + diag + points + region_labels
    
# -----------------------------
# Sidebar controls 
# -----------------------------

# Sidebar controls expose key workflow decisions to the user.
# This supports transparency and human-in-the-loop operation by making
# data loading, case definition, scoring mode, and logging settings explicit.
with st.sidebar: #starts a block where all UI elements inside it are placed in Streamlit's left sidebar
    st.header("Data") #adds "data" section header in sidebar

    data_source = st.radio( #creates radio buttons letting user choose, default is first option
        "Data source",
        ["Upload CSV", "Local file path"],
        index=0,
        key="data_source_radio", #ket gives the widget a unique internal identifier
    )

    uploaded = None #initialize var to hold the uploaded file
    local_path = None #initialize var to hold the local path string

    if data_source == "Upload CSV": #checks what data source option user selected
        uploaded = st.file_uploader("Upload emails CSV", type=["csv"]) #file uploader that accepts CSV files and stores in uploaded
    else:
        local_path = st.text_input( #shows text box where user can type a local file path
            "Local CSV path",
            value="",
            placeholder=r"C:\Users\mikec\OneDrive\Desktop\Doc School\Dissertation Build\datasets\enron_df_clean_full.csv", #shows example path
        )

    st.divider() #visual divider line in sidebar
    st.header("Triage Mode") #adds the next section header
    triage_mode = st.radio( #creates the radio buttons letting user choose between rubric-only or ML mode
        "How should LEDETA prioritize?",
        ["Rubric only (proxy ground truth)", "ML model trained to predict rubric score"],
        index=0,
    )

    st.divider() #visual divider line
    st.header("Queue Ranking") #queue ranking section label
    if triage_mode.startswith("Rubric"): #checks if current mode is rubric
        rank_mode = st.radio( #lets user decide whether to rank by final triage score or by ethics score BEFORE spam penalty
            "Rank cases by",
            ["Triage score (ethics minus spam)", "Ethics score (ignore spam penalty)"],
            index=0,
        )
    else:
        rank_mode = "Triage score (ethics minus spam)"  # ranking is fixed to triage score because model predicts that score

    st.divider() #visual divider line
    st.header("Spam handling") #start the spam handling section
    include_spam_filtered = st.checkbox("Include spam-filtered cases", value=False) #adds a checkbox that lets user keep or excuse cases flagged as spam. default = unchecked which excludes them
    st.caption("Default filters out newsletter/promo-dominant cases with low ethics signal.") #explanation under spam checkbox

    st.divider() #visual divider line
    st.header("Case Definition") #start case def section
    window_days = st.number_input("Window size (days)", min_value=7, max_value=60, value=30, step=1) #shows numeric input box letting user choose case window size, min 7, max 60, default 30

    st.divider() #visual divider line
    st.header("ML Evaluation") #start ML eval section
    if triage_mode.startswith("ML"): #checks current mode is ML
        force_retrain = st.checkbox("Force retrain ML model", value=False) #if in ML mode, provides checkbox to force the model to retrain instead of loading a saved one
        ml_eval_sample_n = 0  # Always evaluate all cases — no sampling
    else: #if not in ML mode, set the variables to harmless defaults
        force_retrain = False
        ml_eval_sample_n = 0

    st.divider() #visual divider line
    st.header("Audit") #start audit  section
    run_id = st.text_input("Run ID (optional)", value="") #add textbox where user can provide a run ID for logging and traceability
    log_run_metrics = st.checkbox("Log runtime metrics to audit log", value=True) #checkbox controlling whether runtime metrics should be written to audit log

run_id = (run_id or "").strip() #after leaving sidebar block, normalize run_id by replacing none with an empty string and stripping extra whitespace


# -----------------------------
# Load emails
# -----------------------------

# Load the email dataset with caching to avoid repeated full reads during
# Streamlit reruns. Only needed columns are loaded to reduce memory use
# and support efficient prototype operation.
@st.cache_data(show_spinner=False) #tell Streamlit to cache output of function below so repeated reruns don't reload same data. showpspinner=false suppresses default laoding spinner
def load_emails(uploaded_file, local_path: str | None) -> pd.DataFrame: #defines function to load email data from soource and return df
    if uploaded_file is None and (not local_path): #if user has no provided file, return empty df with min expected columns
        return pd.DataFrame(columns=["employee", "date", "subject", "body"])

    desired = [ #defines list of columns the app wants to load if they exist. this limits memory use and keeps relevant fields
        "employee", "date", "subject", "body",
        "folder", "body_length", "subject_length",
        "from", "to", "cc", "bcc", "message_id", "file"
    ]

    def _read_header(path_or_buf): #helper func that reads the header row of a CSV to discover which columns are available
        return pd.read_csv(path_or_buf, nrows=0)

    if uploaded_file is not None: #checks whether user uploaded a file
        header = _read_header(uploaded_file) #reads just header row of uploaded file
        uploaded_file.seek(0) #resets file pointer back to beginning so full file can be read next. w/o this pandas might read from end of header row
        cols = [c for c in desired if c in header.columns] #builds list of desired columns that actually exists in file
        df = pd.read_csv(uploaded_file, usecols=cols, low_memory=False) #reads the CSV into a df using only selected columns. low_memory=false asks pandas to do fuller type inference
    else: #user did not uploade file, do local path
        if not os.path.exists(local_path): #if local_path doesnt exist, return empty df instead of crashing
            return pd.DataFrame(columns=["employee", "date", "subject", "body"])

        header = _read_header(local_path) #reads header of local CSV file
        cols = [c for c in desired if c in header.columns] #keeps only relevant columns that are present

        chunks = [] #initialize empty list that will store chunks of CSV
        for chunk in pd.read_csv(local_path, usecols=cols, low_memory=False, chunksize=200_000):
            chunks.append(chunk) #reads local CSV in chunks of 200k rows and then stores each chunk in list
        df = pd.concat(chunks, ignore_index=True) #concatenates all chunks into one df and resets row index

    df["date"] = pd.to_datetime(df["date"], errors="coerce") #converts data column to datetime format. invalid values -> NaT
    return df #return fully loaded df


t0 = time.perf_counter() #start high precision timer for measuring load time
emails = load_emails(uploaded, local_path) #calls your load func and stores email df in emails
t_load = time.perf_counter() - t0 #computes how many seconds the load step took

# Validate that the minimum required columns exist before continuing.
# Early validation prevents invalid inputs from contaminating case construction
# or downstream scoring results.
if emails.empty: #checks whether loaded df is empty
    if data_source == "Local file path" and local_path and (not os.path.exists(local_path)): #checks if reason for emptiness is that local file path doesn't exist
        st.error(f"Local path not found:\n\n`{local_path}`") #error message w/ missing path
    else: #if file empty for another reason
        st.info("Upload a CSV (small) or enter a local file path (large) to begin.") #tell user how to begin
    st.stop() #stops Streamlit script from running any further until valid input supplied

required_cols = {"employee", "date", "subject", "body"} #defines set of columns that must exist for app to work
missing = required_cols - set(emails.columns) #computes which required columns are missing by set substrtaction
if missing: #check whether required columns are missing
    st.error(f"Missing required columns: {sorted(missing)}") #shows an error listing the missing columns 
    st.stop() #stops app to avoid running downstream logic on incomplete data


# -----------------------------
# Build cases (CACHED)
# -----------------------------

# Build case-level units from email records using the dissertation's
# case definition: one employee over one non-overlapping time window.
# Caching reduces repeated computation during interactive analysis.
@st.cache_data(show_spinner=False) #tells Streamlit to cache the output of next function so case-building does not rerun every time the page refreshes unless input changes
def cached_build_cases(emails_df: pd.DataFrame, window_days_i: int): #helper func takes loaded email df and window size in days, then reruns built cases
    return build_cases(emails_df, window_days=window_days_i) #calls imported build_cases function and passes in window size. where raw emails become case objects


t0 = time.perf_counter() #starts timer to measure case building times
cases = cached_build_cases(emails, int(window_days)) #builds cases from email df using selected sidebar window size. ensures input is treated as integer
t_cases = time.perf_counter() - t0 #stores total elapsed time for case building

# Fast O(1) case lookup
case_by_id = {c["case_id"]: c for c in cases} #creates dict where each key is a case_id and each val is full case object. makes retrieval fast when user selects case

st.subheader("Cases") #cases section heading
st.write(f"Built **{len(cases)}** cases from **{emails['employee'].nunique()}** employees.") #shows how many cases were built and how many unique employees in the dataset. 


# -----------------------------
# Score cases (Rubric OR ML)
# -----------------------------

# Score each case using the rubric-based prioritization logic.
# This provides the dissertation's transparent proxy ground truth and
# supports human review without claiming to predict guilt or misconduct directly.
logger = AuditLogger(run_id=run_id or None) #creates an audit logger obj. if the user entered a run ID, uses that, otherwise None

t0 = time.perf_counter() #starts timing scoring or ML section
ml_eval_metrics = None #initialize ML eval metrics variable to None to exists even if rubric mode is used
train_res = None # initialize the ML training result variable to None as well



if triage_mode.startswith("Rubric"): #checks whether traige mode is rubric
    cases_scored = [] #creates empty list to hold scored version of each case
    spam_filtered_count = 0 #starts counter to keep track of how many cases were marked as spam-filtered

    for c in cases: #loops through every case obj built earlier
        res = score_case_rubric_v3(c) #sends current case into rubric scoring function & stores the returned results dict in res

        # HARD SPAM FILTER: drop newsletters/promos unless user opts in
        spam_filtered = bool(res.get("spam_filtered", False)) #looks for spam_filtered value in rubric result. if missing, default to False
        if spam_filtered: #checks whether case was flagged as spam-filtered
            spam_filtered_count += 1 #if yes +1
            if not include_spam_filtered: #if user did not want spam-filtered cases to be included, skip this case entirely
                continue #next loop iter

        triage_score = float(res.get("triage_score", 0.0)) #pulls triage score out of rubric result and converts to float. default = 0.0
        ethics_score = float(res.get("fraud_score", 0.0))  # LEGACY NAME- stores as ethics score. updated conceptually but still reads old key for compatibility
        spam_penalty = float(res.get("spam_penalty", 0.0)) #get spam penalty value from rubric output
        reasons = res.get("reasons", []) #gets list of rubric reasons explaining why case got that score. default = empty list

        c_out = dict(c) #creates a copy of OG case obj as dict so i can add new scoring fields w/o altering OG object
        c_out["triage_score"] = triage_score # adds final triage score to output case dict
        c_out["fraud_score"] = ethics_score        # LEGACY NAME - stores same ethics score under old fraud_score so older parts of sys work
        c_out["ethics_score"] = ethics_score       # stores the score under NEWER clearer name
        c_out["spam_penalty"] = spam_penalty       # adds the spam penalty to case record
        c_out["spam_filtered"] = spam_filtered     #adds true/false spam flag to case record
        c_out["priority_score"] = triage_score     # LEGACY NAME - stores triage score under another genetic key for backward compatibility

        c_out["triage_band"] = pd.cut( #place triage score into one of four fixed bands
            [triage_score], bins=[-1, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Critical"] #low:0-25, med:26-50,high:51-75,crit:76-100
        )[0]
        c_out["fraud_band"] = pd.cut( #LEGACY NAME - old name for compatability, same banding process for ethics score
            [ethics_score], bins=[-1, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Critical"]
        )[0]

        c_out["priority_band"] = c_out["triage_band"] #sets generic priority_band equal to triage band
        c_out["rubric_reasons"] = reasons #stores the rubric explanation reasons on case record
        cases_scored.append(c_out) #adds fully scored case dict to case_scored list

    df_cases = pd.DataFrame(cases_scored) #converts the list of scored case dicts into df

    # ✅ Ensure columns exist / are clean (helps export + filters)
    if not df_cases.empty: #only do cleanup if df has at least 1 row
        df_cases["spam_filtered"] = df_cases.get("spam_filtered", False).fillna(False) #ensure spam_filtered column exists and contains booleans. if column missing get() will supply False and same for fillNA

    if include_spam_filtered: #if user includes spam-filtered cases, display ho many such cases were present
        st.caption(f"Spam-filtered cases included: {spam_filtered_count}")
    else: #if not, tell user how many were removed and how to toggle otherwise
        st.caption(f"Spam-filtered cases removed from queue: {spam_filtered_count} (toggle in sidebar to include)")

    if df_cases.empty: #check whether all cases were filtered out
        st.warning("All cases were filtered out (likely spam/newsletters). Try enabling 'Include spam-filtered cases'.") #shows warning if nothing remains after filtering
        st.stop() #stop app from continuing w/ empty case queue


    if rank_mode.startswith("Ethics"): #checks whether selected ranking is Ethics
        sort_col = "ethics_score" #if so, rank by ethics score
        band_col = "fraud_band" #legacy internal column still maps to ethics severity
    else: #otherwise sort by triage score
        sort_col = "triage_score"
        band_col = "triage_band" #use triage band as severity column

    df_cases = df_cases.sort_values(sort_col, ascending=False) #sort cases from highest score to lowest based on chosen score column
    df_cases["display_score"] = df_cases[sort_col] #creates generic display_score column so rest of UI can work the same way regardless of which score is being shown
    # -----------------------------
    # Band calibration (CRITICAL FIX)
    # -----------------------------
    
# Calibrate display bands using score quantiles so the queue remains
# operationally informative when score distributions are compressed.
# This changes display categories, not the underlying raw score.
    USE_QUANTILE_BANDS = True #turns on quantile-based banding logic. if false, app would use original fixed bands

    CRITICAL_PERCENTILE = 0.97   # top 3% - defines percentile threshold used for crit band.
    HIGH_PERCENTILE = 0.85 #85th percentile
    MED_PERCENTILE = 0.60 #60th percentile
    CRITICAL_FLOOR = 85.0 # defines absolute minimum fora case to stay crit, even if quantile calc would have it placed there

    if USE_QUANTILE_BANDS and len(df_cases) >= 50: #only use quantile banding if feature is enabled and at least 50 cases. avoids unstable quantiles on tiny datasets
        s = pd.to_numeric(df_cases["display_score"], errors="coerce").dropna() #converts display score column to numeric, coercing invalid values to missing, removes missing. clean numeric series for quantile calc

        if len(s) >= 50: #checks still at least 50 valid numeric scores after cleaning
            q_med = float(s.quantile(MED_PERCENTILE)) #calcs numeric score val at 60th
            q_high = float(s.quantile(HIGH_PERCENTILE)) #calcs numeric score val at 85th
            q_crit = float(s.quantile(CRITICAL_PERCENTILE)) #calcs numeric score val at 97th

            # enforce monotonic cutoffs (cutoffs never go backwards)
            cuts = np.maximum.accumulate([q_med, q_high, q_crit]) #ensure three quantile cutoffs are nondecreasing. if any threshold came out lower, it will fix
            q_med, q_high, q_crit = cuts.tolist() #unpacks corrected cutoff vals back into three vars

            df_cases["display_band"] = pd.cut( #creates dislay_band column by slicing the display_score values into 4 quantile-based intervals
                df_cases["display_score"],
                bins=[-np.inf, q_med, q_high, q_crit, np.inf],#-inf-q_med = low,  q_med-q_high=med, q_high-q_crit=high, q_crit-+inf=crit
                labels=["Low", "Medium", "High", "Critical"],
                include_lowest=True,#ensures smallest values are included
            ).astype(str) #converts categorical labels to strings

            # hard floor: demote weak Criticals
            crit_mask = ( #creates boolean mask identifying cases that were labeled critical by the quantile rule but have a score below hard floor of 85
                (df_cases["display_band"] == "Critical")
                & (df_cases["display_score"] < CRITICAL_FLOOR)
            )
            df_cases.loc[crit_mask, "display_band"] = "High" #changes the weak critical cases to high. .loc used to modify only the rows matching the mask

            st.caption(
                f"Severity calibration: Medium≥{q_med:.1f}, "
                f"High≥{q_high:.1f}, "
                f"Critical≥max({q_crit:.1f}, {CRITICAL_FLOOR:.1f})"
            )
    else: #if qunatile banding is turned off or there are too few cases, use original fixed band column instead
        # fallback to original fixed bands
        df_cases["display_band"] = df_cases[band_col]

# In ML mode, train or load a supervised model that predicts the
# rubric-derived case score. The model is evaluated against rubric scores
# because the dissertation's target variable is the proxy prioritization score.
else: #if app not in rubric mode, enter ML mode branch
    train_res = train_or_load_model(cases, force_retrain=bool(force_retrain)) #calls training/loading func, passes in all cases, and whether user forced retraining
    sample_n = None  # Always use all cases

    # --- Show what ML model is actually being used (LightGBM vs fallback) ---
    model_kind = None #intializes the model kind variable
    if hasattr(train_res, "metrics") and isinstance(train_res.metrics, dict): #checks whether train_res has metrics attribute and whether it is a dict
        model_kind = train_res.metrics.get("model_kind", None) #if metrics exist, pull out the model_kind val from them

    st.info( #tells user: which model loaded, whether it retrained, and what file path (TRANSPARENCY)
        f"ML model loaded: **{model_kind or 'unknown'}** | "
        f"retrained: **{bool(train_res.metrics.get('retrained', False)) if hasattr(train_res, 'metrics') else 'unknown'}** | "
        f"path: `{getattr(train_res, 'model_path', '')}`"
    )

    df_eval, ml_eval_metrics = predict_cases_with_rubric_comparison( #calls your prediction/comparison func
        train_res, #df_eval = the case df including rubric and ML scores
        cases, #ml_eval_metrics = summary metrics like MAE, RMSE, R^2, and sample size
        sample_n=sample_n,
    )

    df_cases = df_eval.sort_values("ml_score", ascending=False) #sorts evaluated cases by ML score high -> low
    df_cases["display_score"] = df_cases["ml_score"] #creates generic display_score column
    df_cases["display_band"] = df_cases["ml_band"] #creates generic display_band column

t_score = time.perf_counter() - t0 #calculates total time spent in scoring section, for rubric or ML


# -----------------------------
# Investigator filters + export
# -----------------------------

# Apply user-controlled filters so the prioritized queue can be narrowed
# by severity, employee, and date range. This supports practical review workflows
# rather than one-size-fits-all output.
t_filter_start = time.perf_counter() #starts timming the filter section

st.sidebar.divider() #divider
st.sidebar.header("Investigator Filters") #header in sidebar

# LATER ADDITION: toggle to hide spam/newsletter-dominant cases (rubric flag)
hide_spam_filtered = st.sidebar.checkbox( #creates sidebar checkbox (default True) meaning spam/newsletter dominant cases are hidden unless user puts them back one.
    "Hide spam/newsletter-dominant cases",
    value=True,
    help="Uses rubric spam_filtered flag to remove obvious newsletters/promos by default." #explanation
)

df_cases["window_start"] = pd.to_datetime(df_cases["window_start"], errors="coerce") #converts window_start to datetime so date filtering works
df_cases["window_end"] = pd.to_datetime(df_cases["window_end"], errors="coerce") #same for window_end

all_bands = ["Low", "Medium", "High", "Critical"] #defines 4 possible options for filter widget
selected_bands = st.sidebar.multiselect("Severity", options=all_bands, default=all_bands) #crreates a multiselect widget allowing user to include and combo of 4 bands. default all are selected

employees_sorted = sorted(df_cases["employee"].dropna().unique().tolist()) #builds sorted list of unique employee names, exclusing missing. populates employee filter
selected_employees = st.sidebar.multiselect("Employees", options=employees_sorted, default=[]) #creates multiselect widget for employees. default is empty, unless user selects

min_date = df_cases["window_start"].min() #find earliest case window start date in current case df
max_date = df_cases["window_start"].max() #find latest case window start date

if pd.isna(min_date) or pd.isna(max_date): #if either date is missing, app disables date filtering
    date_range = None
else: #otherwise, a date-range input widget using earliest and latest case dates at default selected interval
    date_range = st.sidebar.date_input(
        "Case window start date range",
        value=(min_date.date(), max_date.date()),
    )

df_filtered = df_cases.copy() #a copy of full case df so filters can be applied w/o changing OG 

# Apply filters (bands, employees) - filter by selected bands and employees
if selected_bands:#if user selected any bands, keep only rows whose display_band is in the selected list
    df_filtered = df_filtered[df_filtered["display_band"].astype(str).isin(selected_bands)] #isin checks membership
if selected_employees: #if user selected any employees, keep only rows for those employees
    df_filtered = df_filtered[df_filtered["employee"].isin(selected_employees)]

# Apply date range filter
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range): #checks that date_range is a proper 2-val tuple and that both dates are present
    start_d, end_d = date_range #unpacks date range tuple into start and end date variable
    start_ts = pd.to_datetime(start_d) #converts start date to pds timestamps
    end_ts = pd.to_datetime(end_d) + pd.Timedelta(days=1) #converts end date to a timestamp, then adds one day so the filter behaves as inclusive of sleected end day

    ws = df_filtered["window_start"] #stores window_start column in shorter variable ws
    if getattr(ws.dt, "tz", None) is not None: #checks whether datetime coolumn has timezone info. getattr avoids errors if attribute is absent
        tz = ws.dt.tz #if there is a timezone, save it
        start_ts = start_ts.tz_localize(tz) #localizes start and end timestamps to same timezone as case data so comparisons work
        end_ts = end_ts.tz_localize(tz)

    df_filtered = df_filtered[ #keeps only rows whose window_start is on or after selected start date and strictly before the day after selected end time. inclusive
        (df_filtered["window_start"] >= start_ts) &
        (df_filtered["window_start"] < end_ts)
    ]

# NEW: spam/newsletter filter AFTER all other filters
if hide_spam_filtered and "spam_filtered" in df_filtered.columns: #checks user wants spam hidden and that df actually has a spam_filtered column
    df_filtered = df_filtered[~df_filtered["spam_filtered"].fillna(False)] #keeps only rows where spam_filtered is not true. missing values treated as false

t_filter = time.perf_counter() - t_filter_start #meausre how long filtering took

if df_filtered.empty:#checks if filtering removed all cases
    st.warning("No cases match your filters. Try expanding bands, employee list, date range, or uncheck 'Hide spam'.") #warning on how to broaden filters if no cases remain
    st.stop() #stops execution so app doeos not display empty outputs

# Export
base_export_cols = ["case_id", "employee", "window_start", "window_end", "n_emails"] #defines columns always included in export, regardless of mode
if triage_mode.startswith("Rubric"): #checks app is in rubric for right export columns
    wanted_export = base_export_cols + ["triage_score", "triage_band", "ethics_score", "spam_penalty", "spam_filtered"] #rubric mode columns
else:
    wanted_export = base_export_cols + ["ml_score", "ml_band", "rubric_score", "rubric_band", "abs_error"] #ML mode columns

export_cols = [c for c in wanted_export if c in df_filtered.columns] #keep only columns that actually exist in filtered df, prevents errors if optional columns are missing

csv_bytes = df_filtered[export_cols].to_csv(index=False).encode("utf-8") #converts filtered df to CSV string w/o row indices, then encodes as UTF-8 bytes for download
st.sidebar.download_button( #creates sidebar download button that allows user downloade the filtered queue as CSV
    label="Download filtered queue (CSV)",
    data=csv_bytes,
    file_name="ledeta_ranked_queue_filtered.csv",
    mime="text/csv",
)

# -----------------------------
# Runtime metrics (top)
# -----------------------------

# Capture and display runtime metrics so the prototype can support
# efficiency benchmarking alongside predictive validity.
n_emails_total = int(len(emails)) #counts total # of email rows loaded converts to integer
n_cases_total = int(len(cases)) #counts total # of built cases
cases_per_sec = (n_cases_total / t_cases) if t_cases > 0 else 0.0 #calculates how many cases were built per second. if t_cases is zero, returns 0.0 to avoid division by 0
emails_per_sec = (n_emails_total / t_load) if t_load > 0 else 0.0 #calculates how many emails per second were processed during load. if t_cases is zero, returns 0.0 to avoid division by 0

score_label = "Score/model time (s)" if triage_mode.startswith("Rubric") else "ML train/predict time (s)" #set label text for score-time metric. 

with st.expander("⏱️ Runtime metrics", expanded=True): #creates an expandable UI section for runtime metrics
    m1, m2, m3, m4 = st.columns(4) #creates 4 equal-width columns to lay out first row of metrics
    m1.metric("Load time (s)", f"{t_load:.2f}") #displays data load time rounded to two decimal places
    m2.metric("Case build time (s)", f"{t_cases:.2f}") #displays case build time
    m3.metric(score_label, f"{t_score:.2f}") #displays either rubric scoring time or ML train/predict time
    m4.metric("Filter time (s)", f"{t_filter:.2f}") #display filter time

    s1, s2, s3 = st.columns(3) #creates three more columns for second row of metrics
    s1.metric("Emails processed", f"{n_emails_total:,}") #show total # of email rows processed, comma thousands separator
    s2.metric("Cases built", f"{n_cases_total:,}") #total # of cases built
    s3.metric("Cases/sec (build)", f"{cases_per_sec:.2f}") #show how many cases per second built
    
    if triage_mode.startswith("ML") and train_res is not None: #checks whether app is in ML mode and whether training results exist
        mk = None #initializes temp variables to hold ML model kind
        if hasattr(train_res, "metrics") and isinstance(train_res.metrics, dict): #if training result has a metric dict, get model_kind value
            mk = train_res.metrics.get("model_kind", None)
        st.caption( #displays caption showing which model was used and whether it was retrained
            f"ML model kind: {mk or 'unknown'} | "
            f"retrained: {train_res.metrics.get('retrained', False) if hasattr(train_res, 'metrics') else 'unknown'}"
        )
    

    st.caption(f"Emails/sec (load stage): {emails_per_sec:.2f}") #displays email loading throughput


# -----------------------------
# Audit logging: run metrics
# -----------------------------

# Log run-level metadata for traceability and governance review.
# This preserves how the system was configured and how it performed.
if log_run_metrics: #checks whether the user wants runtime metrics logged
    run_payload = { #starts building the dictionary that will be written to the audit
        "event": "run_metrics", #labels this audit record as a run-metrics event
        "ts": _now_iso(), #adds the current timestamp using helper func
        "run_id": run_id or None, #stores the run ID if it exists, otherwise None
        "data_source": data_source, #stores whether the data came from upload or local path
        "window_days": int(window_days), #stores selected case window size
        "triage_mode": triage_mode, #stores whether the run used rubric or ML mode
        "n_emails": n_emails_total, #stores total # of emails loaded
        "n_cases": n_cases_total, #store total # of built cases
        "t_load_s": float(t_load), #stores load time in seconds
        "t_case_build_s": float(t_cases), #stores case-build time
        "t_score_s": float(t_score), #stores scoring or ML time
        "t_filter_s": float(t_filter), #stores filter time
        "random_state": GLOBAL_RANDOM_STATE, #stores the random seed used for reproducibility
        "include_spam_filtered": bool(include_spam_filtered), #stores whether spam-filtered cases were allowed into queue
    }
    if ml_eval_metrics is not None: #if ML metrics exist, add theem to audit payload
        run_payload["ml_eval_metrics"] = ml_eval_metrics #stores full ML evaluation metrics dictionary in payload
        if train_res is not None: #if training results exist, also store the file path of the model used. getattr safely returns None if model_path is missing
            run_payload["ml_model_path"] = getattr(train_res, "model_path", None)

    _append_jsonl(logger.path, run_payload) #writes the payload as a JSON line into audit log file


# -----------------------------
# Display table + summary chart (filtered)
# -----------------------------

# Display the filtered case queue and a visual summary of severity counts
# so reviewers can inspect both individual cases and overall distribution.
col1, col2 = st.columns([2, 1]) #create 2 columns, left one twice as wide as right one. table goes on left, summary stats on right

with col1: #starts the left column block
    if triage_mode.startswith("Rubric"): #checks if rubric mode to choose correct display columns
        df_filtered["severity"] = df_filtered["display_band"].astype(str)

        wanted_display = [
            "case_id", "employee", "window_start", "window_end", "n_emails",
            "display_score", "severity",
            "triage_score", "triage_band",
            "ethics_score",
            "spam_penalty", "spam_filtered",
        ]
    else: #or the desired table columns for ML mode
        df_filtered["severity"] = df_filtered["display_band"].astype(str)

        wanted_display = ["case_id", "employee", "window_start", "window_end", "n_emails", "ml_score", "severity", "rubric_score", "abs_error"]

    display_cols = [c for c in wanted_display if c in df_filtered.columns]

    display_df = df_filtered[display_cols].copy().rename(columns={
        "display_score": "Severity score",
        "severity": "Severity",
        "triage_score": "Triage score",
        "triage_band": "Triage severity",
        "ml_score": "ML score",
        "ml_band": "ML severity",
        "rubric_score": "Rubric score",
        "rubric_band": "Rubric severity",
        "ethics_score": "Ethics score",
        "spam_penalty": "Spam penalty",
        "spam_filtered": "Spam filtered",
        "case_id": "Case ID",
        "window_start": "Window start",
        "window_end": "Window end",
        "n_emails": "Emails",
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col2: #starts right summary column
    st.metric("Top score", f"{float(df_filtered['display_score'].max()):.2f}")
    st.metric("Median score", f"{float(df_filtered['display_score'].median()):.2f}")
    st.altair_chart(_band_counts_chart(df_filtered, triage_mode), use_container_width=True) #builds and shows the band-count chart using your helper function

    if triage_mode.startswith("ML"): #if in ML mode, shows a color legend caption for the grouped band chart
        st.caption("Color key: **Rubric = blue**, **ML = red**")


# -----------------------------
# ML vs Rubric evaluation panel (ML mode only)
# -----------------------------

# Show predictive validity metrics and error analysis so users can assess
# how closely ML predictions align with rubric-derived scores.
if (not triage_mode.startswith("Rubric")) and (ml_eval_metrics is not None): #only show this eval panel if app is in ML mode and ML metrics were produced
    st.divider() #divider added
    st.subheader("ML vs Rubric (ground truth) — Evaluation") #section title

    a, b, c, d = st.columns(4) #creates 4 columns for metric cards
    a.metric("MAE", f"{ml_eval_metrics['mae']:.2f}") #displays MAE (mean absolute error rounded to 2 dec)
    b.metric("RMSE", f"{ml_eval_metrics['rmse']:.2f}") #displays root mean squared (RMSE) rounded 2
    c.metric("R²", f"{ml_eval_metrics['r2']:.3f}") #displays R-squared to 3 dec
    d.metric("N evaluated", f"{ml_eval_metrics['n_eval']:,}") #displays how many cases were evaluated

    chart_df = df_cases[["case_id", "employee", "rubric_score", "ml_score", "abs_error"]].dropna() #builds a smaller df containing columns needed for scatterplot and remove missing value rows
    sc = _ml_scatter_chart(chart_df) #generates the ML vs rubric scatter chart using helper func.
    if sc is not None: #checks that helper func actually returned a chart
        st.altair_chart(sc, use_container_width=True) #displays the scatter chart
        st.caption(
            f"Color key: **ML points = red**. Diagonal reference line = **rubric agreement (blue)**. "
            f"Vertical cutoff lines at 25, 50, and 75 indicate the Low, Medium, High, and Critical score regions. "
            f"A small jitter is applied to reduce overlap; true values are preserved in tooltips. "
            f"Scatterplot displays all {ml_eval_metrics['n_eval']:,} evaluated cases. "
            f"Total cases built in this run: {n_cases_total:,}."
        )

    st.markdown("**Largest ML errors (top 10)**") #label for error table
    if "abs_error" in df_cases.columns: #checks that df contains absolute error column
        st.dataframe( #sorts cases by absolute error from large -> small, selects relevant columns
            df_cases.sort_values("abs_error", ascending=False)[
                [c for c in ["case_id", "employee", "rubric_score", "ml_score", "abs_error", "window_start", "window_end", "n_emails"] if c in df_cases.columns]
            ].head(10), #keeps top 10 
            use_container_width=True, 
            hide_index=True,
        ) #displays them in table and helps to inspect where model disagreed most with the rubric


# -----------------------------
# Case drill-down
# -----------------------------

# Allow the user to inspect the raw emails behind a prioritized case.
# This supports human verification and prevents the score from being treated
# as a standalone or final decision.
st.divider() #divider
st.subheader("Case drill-down") #section header

selected = st.selectbox("Select a case", df_filtered["case_id"].tolist()) #creates a dropdown containing case IDs of all currently filtered cases. current ID is in selected
case_row = df_filtered[df_filtered["case_id"] == selected].iloc[0].to_dict() #finds the row in the filtered df whose case_id matches the selected value then takes 1st matching row and converts to dict

if triage_mode.startswith("Rubric"): #checks whether rubric mode is selected
    st.markdown( #displays a 1-line summary of selected rubric-scored case
        f"**Ethics score:** {float(case_row.get('ethics_score', case_row.get('fraud_score', 0.0))):.2f} | " #includes ethics score FALLS BACK TO LEGACY FRAUD_SCORE IF NEEDED
        f"**Triage score:** {float(case_row.get('triage_score', 0.0)):.2f} | " #includes triage score
        f"**Spam penalty:** {float(case_row.get('spam_penalty', 0.0)):.2f} | " #includes spam penalty
        f"**Spam filtered:** {bool(case_row.get('spam_filtered', False))}" # includes spam flag
    )

# O(1) lookup
case_obj = case_by_id[selected] #looks up full OG case obj by its case ID

st.markdown("### Emails in this case") #heading above email list for selected case

@st.cache_data(show_spinner=False) #caches the helper func below so email header extracction does not rerun unnecessarily
def cached_case_email_headers(case_id: str, case_obj_dict: dict) -> pd.DataFrame: #defines func to take case ID and case obj dict, then return a df of email header info
    rows = [] #empty list that will hold one dict per email
    for idx, e in enumerate(case_obj_dict.get("emails", [])): #loops through emails stored inside the case obj. enumerate gives both numeric index and email dict. if emails key is missing defaults to empty list
        rows.append({  #creates small dict for each email 
            "_idx": idx, #email's original position in case
            "date": e.get("date"), 
            "folder": e.get("folder", ""),
            "from": e.get("from", ""),
            "to": e.get("to", ""),
            "subject": e.get("subject", ""),
        })
    df = pd.DataFrame(rows) #converts list of email heaader dicts into a df
    if not df.empty: #only do the next cleanup if header df has rows
        df["date"] = pd.to_datetime(df["date"], errors="coerce") #converts email date column to datetime values. invalid ones become missing
        df = df.sort_values("date", ascending=False) #sorts the emails so the newest ones appear first
    return df #returns the email header df


case_emails = case_obj.get("emails", []) #pulls list of raw email dicts out of the full case obj. defaults to an empty list if missing
if not case_emails: #checks whether the selected case actually contains any emails
    st.info("No emails were attached to this case object. (Check build_cases: it should store case['emails'].)") #if no emails, show message
else: #if the case does have emails continue with drill-down UI
    q = st.text_input("Search within case emails (subject only — fast)", value="") #creates a text box so the user can search in the case's emails by subject line only. default to blank

    df_case_emails = cached_case_email_headers(selected, case_obj) #builds the email-header df for the selected case using the cached helper

    if q.strip() and (not df_case_emails.empty): #checks whether search box contains non-whitespace text and email header df isnt empty
        qq = q.strip().lower() #strips extra spaces from the query and lowercases it for case-insensitive matching
        df_case_emails = df_case_emails[ #filters email header dfs to only emails whose subject contains the search term (case-insensitivity)
            df_case_emails["subject"].fillna("").str.lower().str.contains(qq) #missing subjects treated as empty strings
        ]

    st.dataframe( #displays the filtered email header table
        df_case_emails[["date", "folder", "from", "to", "subject"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### View full email") #heading above full-email viewer
    if len(df_case_emails): #checks if there are any email rows left to view
        def _safe_str(x): #defines a small helper func that converts None to an empty string 
            return "" if x is None else str(x) #everything else to a string. prevent formatting issues in dropdown labels

        email_idx = st.selectbox( #creates a dropdown for choosing 1 email from displayed email list
            "Select email",
            options=list(range(len(df_case_emails))), #options are nmeric positions 0 through N-1
            format_func=lambda i: f"{df_case_emails.iloc[i]['date']} — {_safe_str(df_case_emails.iloc[i]['subject'])[:80]}", #user sees formatted labels showing the date and first 80 chars of subject line
        )

        header_row = df_case_emails.iloc[email_idx].to_dict() #gets chosen row from the email header df and converts to dict
        raw_email = case_emails[int(header_row["_idx"])] #uses original _idx value to retrieve corresponding FULL RAW email from OG email list
        #IMPORTANT BECAUSE DF MAY HAVE BEEN SORTED OR FILTER, SO CURRENT POS MAY NOT MATCH OG EMAIL LIST ORDER
        st.markdown(f"**Date:** {header_row.get('date')}") #displays select email's date
        st.markdown(f"**Folder:** {header_row.get('folder')}") #displays the folder
        st.markdown(f"**From:** {header_row.get('from')}") #displays sender
        st.markdown(f"**To:** {header_row.get('to')}") #displays recipient
        st.markdown(f"**Subject:** {header_row.get('subject')}") #displays subject line
        st.text_area("Body", value=str(raw_email.get("body", "") or ""), height=260) #displays full email body in non-editable text
        #THIS IS IN 260 PIXELS AND SAFELY CONVERTS MISSING VALUES TO EMPTY STRINGS
        
# -----------------------------
# Explanation + audit (button-controlled + cached) - original design was too resource intensive with ALL explanations
# -----------------------------

# Generate case-level explanations on demand to balance transparency with
# computational efficiency. Explanations include summary text, top drivers,
# and evidence snippets to support human review.
case_row_for_explain = dict(case_row) #create copy of selected case row dict so it can be adjusted for explanation gen w/o changing displayed case row
if not triage_mode.startswith("Rubric"): #checks if app is in ML mode
    case_row_for_explain["priority_score"] = float(case_row_for_explain.get("ml_score", 0.0)) #if in ML mode, overwrite priority_score with ML score so explanation uses ML outout as current priority score
    case_row_for_explain["priority_band"] = case_row_for_explain.get("ml_band", "Unknown") #overwrite priority band with ML band. keeps explanation aligned with what user is seeing in ML mode

@st.cache_data(show_spinner=False) #caches the explanation helper below
def cached_explain(case_id: str, case_obj_dict: dict, row_for_explain: dict): #defines helper func that takes selected case + row data and returns an explanation
    return explain_case(case_obj_dict, row_for_explain) #calls imported explain_case func and returns result

st.divider() #add divider
st.subheader("Explanation") #add section title

do_explain = st.button("Generate explanation for this case", type="primary") #creates a prmary-action button. do_explain becomes true if user clicks

if do_explain: #checks if explanation button was clicked
    t0 = time.perf_counter() #starts timing explanation generation
    with st.spinner("Generating explanation..."): #shows a spinner while explanation generates
        explanation = cached_explain(selected, case_obj, case_row_for_explain) #generates explanation using cached helper + stores result in explanation val
    t_explain = time.perf_counter() - t0 #measures how long explanation gen took

    logger.log_case(case_obj, case_row_for_explain, explanation) #logs selected case, scoring row using for explanation and explanation payload using your audit logger

    if log_run_metrics: #checks if runtime logging is enabled
        _append_jsonl(logger.path, { #writes a JSONL audit record that describes the explanation_generated event
            "event": "case_explain_metrics",
            "ts": _now_iso(), #time stamp
            "run_id": run_id or None, #run ID
            "case_id": selected, #case ID
            "t_explain_s": float(t_explain), #elapsed explanation time
        })

    c1, c2 = st.columns([1, 1]) #creates 2 equal width columns for explaantion display

    with c1: #starts the left explanation column
        st.markdown("### Summary") #summary heading
        st.write(explanation.get("summary", "")) #displays explanation summary text. default to empty if missing
        st.markdown("**Top drivers**") #label for top-driver table
        st.dataframe(pd.DataFrame(explanation.get("top_features", [])), use_container_width=True, hide_index=True) #converts top_features list into a df and displays it. if missing use an empty list

    with c2: #starts right explanation column
        st.markdown("### Evidence snippets") #heading added
        for snip in (explanation.get("snippets", [])[:6]): #loops through first 6 snippet from explanation output. if none, use empty list
            title = f"{snip.get('date', '')} — {snip.get('subject', '')}" #creates a title string for each snippet using data and subject
            if snip.get("folder"): #if snippet includes folder name
                title += f"  [{snip['folder']}]" #append in brackets to the title
            with st.expander(title, expanded=False): #creates a collapsible expander for snippet, closed by default
                st.write(snip.get("excerpt", "")) #displays the snippet exerpt text inside expander

else: #if the button has not been click, show an informational note as to why the explanation must be generated via button
    st.info("Explanation generation is expensive. Click **Generate explanation** when you’re ready.")

# Expose the audit log location so the system's traceability mechanism
# remains visible to the user.
st.markdown("### Audit log") #audit log heading
st.code(logger.path, language="text") #displays file path of audit log in code block
st.caption("Each run writes JSONL entries with timestamp, score, features, and explanation payload.") #a caption explaining what kind of info is written to audit log
