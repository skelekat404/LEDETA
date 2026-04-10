# LEDETA - Law Enforcement Digital Evidence Triage Assistant

> A dissertation prototype for **case-level** triage of text-based digital evidence (emails).  
> A **case** = all emails for one employee within a 30-day window.

---

## 🚀 Quick Start

```bash
python -m venv .venv

# Activate (macOS/Linux):
source .venv/bin/activate

# Activate (Windows):
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Data Sources

| Option | Description |
|--------|-------------|
| **Built-in demo** | Synthetic dataset included in `/datasets/` - recommended for first run |
| **Upload CSV** | Upload your own file with at least: `employee`, `date`, `subject`, `body` |
| **Enron full dataset** | [Download via Dropbox](https://www.dropbox.com/scl/fo/0f7ayl1fj8f0wzsoyq3s2/AEPwwA9NZlitAlCq6OYkaYw?rlkey=n4dzsedpyn9lqw2dwqtreipdk&st=muaum8nx&dl=0) — place in `/datasets/` folder |

---

## 🗂️ File Overview

| File / Folder | Purpose |
|---------------|---------|
| `app.py` | Main Streamlit app - loads data, builds cases, scores, filters, and renders the full triage UI |
| `ledeta/case_builder.py` | Aggregates raw email rows into case objects (employee + 30-day window) |
| `ledeta/rubric.py` | Rubric-based prioritization engine - produces proxy ground-truth triage scores |
| `ledeta/model.py` | Trains or loads LightGBM ML model; generates predictions and rubric comparisons |
| `ledeta/features.py` | Feature extraction: volume/communication metrics, keyword signals, TF-IDF text features |
| `ledeta/explain.py` | Generates per-case explanation output: summary text, top drivers, evidence snippets |
| `ledeta/audit.py` | Audit logger - writes structured JSONL records for each scoring/explanation run |
| `tests/run_rq1_predictive_validity.py` | RQ1 test: predictive validity evaluation |
| `tests/run_rq2b_inference_only.py` | RQ2b test: inference-only performance benchmark |
| `tests/run_rq_kfold_window_validation.py` | 5-fold cross-validation - validates emails don't bleed across windows |
| `datasets/` | Holds the synthetic demo dataset (and Enron if downloaded) |
| `ledeta_models/` | Serialized trained models (`.joblib`) |
| `EDA.ipynb` | Exploratory data analysis notebook |
| `audit_log.jsonl` | Running audit log written during app use |
| `rq1_results.csv`, `rq2_runs.csv`, etc. | Stored RQ output files |

---

## ⚙️ What LEDETA Does

- **Case construction** - groups emails by employee and 30-day non-overlapping windows
- **Rubric scoring** - transparent, rule-based priority score (proxy ground truth)
- **ML-assisted mode** - LightGBM model trained to predict rubric scores
- **Severity banding** - quantile-calibrated Low / Medium / High / Critical labels
- **Explainability** - per-case driver scores and evidence snippet surfacing
- **Investigator filters** - filter queue by severity, employee, and date range
- **Audit logging** - every run writes traceable JSONL records

---

## 📄 License

This project is licensed under the **MIT License** - see [`LICENSE`](LICENSE) for details.
