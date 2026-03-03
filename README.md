# LEDETA (MVP) — Law Enforcement Digital Evidence Triage Assistant

This repo is a dissertation prototype for **case-level** triage of text-based evidence (emails).
A "case" is defined as **all emails for one employee in a 30-day window**.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Data expectations (flexible)
This MVP supports two input modes:
1) Upload a CSV with at least: `employee`, `date`, `subject`, `body`
2) Use the built-in demo dataset toggle (synthetic) to validate the app works

If you want to use Enron, convert it into the above CSV schema (one row per email).
Ingestion is modular so we can add Avocado later.

## What this MVP includes
- Case builder: employee + 30-day window aggregation
- Feature extraction:
  - volume/communication metrics
  - keyword indicators (fraud/financial)
  - TF-IDF text features
- Priority rubric (proxy ground truth) + optional supervised model
- Explainability:
  - drivers + evidence snippets
- Audit log (JSONL) for each scoring run

## (NEEDS TO BE UPDATED)

- ++ ADD A DROP BOX LINK WITH ALL DATA FILES

- ++ ADD RQ FILE EXPLANATIONS

- ++ ADD EDA FILE EXPLANATION

- ++ ADD RQ OUTPUT EXPLANATIONS
