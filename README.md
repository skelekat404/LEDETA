# LEDETA — Law Enforcement Digital Evidence Triage Assistant

> A dissertation prototype for **case-level** triage of text-based digital evidence (emails).  
> A **case** = all emails for one employee within a non-overlapping 30-day window.

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
| **Built-in demo** | Synthetic dataset included in `/datasets/` — recommended for first run |
| **Upload CSV** | Upload your own file with at least: `employee`, `date`, `subject`, `body` |
| **Enron full dataset** | [Download via Dropbox](https://www.dropbox.com/scl/fo/0f7ayl1fj8f0wzsoyq3s2/AEPwwA9NZlitAlCq6OYkaYw?rlkey=n4dzsedpyn9lqw2dwqtreipdk&st=muaum8nx&dl=0) — place in `/datasets/` folder |

---

## 🗂️ File Overview

| File / Folder | Purpose |
|---------------|---------|
| `app.py` | Main Streamlit app — loads data, builds cases, scores, filters, and renders the full triage UI |
| `ledeta/case_builder.py` | Groups raw email rows into case objects by employee using non-overlapping 30-day windows |
| `ledeta/rubric.py` | Rubric-based prioritization engine — computes ethics score, applies spam penalty, enforces a critical rarity gate requiring hits on at least two of three top misconduct categories before a score can exceed 75 |
| `ledeta/model.py` | Trains or loads a LightGBM regression pipeline (with median imputation) targeting rubric-derived triage scores; generates ML predictions and rubric comparison metrics (MAE, RMSE, R²) |
| `ledeta/features.py` | Extracts numeric case features: email volume, temporal burst, keyword hit counts across seven ethics signal groups (secrecy, approval bypass, record tampering, vague expenses, distancing language, money operations, bank changes), spam/marketing signals, recipient breadth, and external domain ratios |
| `ledeta/explain.py` | Generates per-case explanation output: summary text, top feature drivers, and evidence snippets |
| `ledeta/audit.py` | Audit logger — writes structured JSONL records for each scored and explained case |
| `tests/run_rq1_predictive_validity.py` | RQ1: predictive validity evaluation (MAE, RMSE, R², Wilcoxon signed-rank, bootstrap CI) |
| `tests/run_rq2b_inference_only.py` | RQ2b: inference-only runtime and throughput benchmark |
| `tests/run_rq_kfold_window_validation.py` | 5-fold stratified cross-validation — validates that cases do not bleed across 30-day windows |
| `datasets/` | Holds the synthetic demo dataset (and Enron full dataset if downloaded) |
| `ledeta_models/` | Serialized trained LightGBM model (`.joblib`) |
| `EDA.ipynb` | Exploratory data analysis notebook |
| `audit_log.jsonl` | Running audit log written during app use |
| `rq1_results.csv`, `rq2_runs.csv`, etc. | Stored RQ output files |

---

## ⚙️ What LEDETA Does

- **Case construction** — groups emails by employee into non-overlapping 30-day windows; each case stores email headers, excerpted body text (800 chars), and metadata
- **Feature extraction** — counts keyword hits across seven ethics signal groups, computes spam signal from unsubscribe/URL/marketing language, calculates temporal burst (max emails in a single day), and measures communication breadth (recipient counts, external domain ratios)
- **Rubric scoring** — weights ethics signals (secrecy ×3.2, bypass ×2.8, tampering ×2.6, urgency, distancing, money ops), applies a spam penalty (up to 45 points), saturates scores to 0–100 via an exponential function, and enforces a critical rarity gate
- **ML-assisted mode** — LightGBM regressor trained on rubric-derived scores using an 80/20 split with median imputation; predictions clipped to 0–100
- **Severity banding** — quantile-calibrated Low / Medium / High / Critical labels with a hard floor preventing weak cases from reaching Critical
- **Explainability** — per-case driver scores and evidence snippet surfacing to support human review
- **Investigator filters** — filter queue by severity band, employee, and case window date range
- **Audit logging** — every scored and explained case writes a traceable JSONL record with timestamp, run ID, score, band, and explanation payload

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.
