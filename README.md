# LEDETA - Law Enforcement Digital Evidence Triage Assistant

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
This prototype supports two input modes: file path pasting or file upload.
1) Upload your own CSV with at least: `employee`, `date`, `subject`, `body` (not recommended since we have demo data)
2) Use the built-in demo dataset in the repo (synthetic) to validate the app works (recommended, varying sizes)
3) Use the Enron cleaned up dataset (download and move to 'datatset' folder in project directory)
   https://www.dropbox.com/scl/fo/0f7ayl1fj8f0wzsoyq3s2/AEPwwA9NZlitAlCq6OYkaYw?rlkey=n4dzsedpyn9lqw2dwqtreipdk&st=muaum8nx&dl=0

## What this includes
- Case builder: employee + 30-day window aggregation
- Feature extraction:
  - volume/communication metrics
  - keyword indicators (misconduct)
  - TF-IDF text features
- Priority rubric (proxy ground truth) + optional supervised model
- Explainability:
  - drivers + evidence snippets
- Audit log (JSONL) for each scoring run

## (NEEDS TO BE UPDATED RIGHT BEFORE FINAL SUBMISSION - APRIL 10TH)

- ++ ADD A DROP BOX LINK WITH BIG ENRON DATA FILE - ✅

- ++ ADD APP FILE EXPLANATIONS - ✅

- ++ ADD RQ 1 FILE EXPLANATIONS - ✅

- ++ ADD RQ 2 FILE EXPLANATIONS -⏳

- ++ ADD EDA FILE EXPLANATION - ⏳
  
- ++ ADD RQ OUTPUT EXPLANATIONS - ⏳

- ++ ADD CASE BUILDER FILE EXPLANATIONS - ⏳

- ++ ADD EXPLAIN FILE EXPLANATIONS - ⏳

- ++ ADD FEATURES FILE EXPLANATIONS -⏳

- ++ ADD MODEL FILE EXPLANATIONS - ⏳

- ++ ADD RUBRIC FILE EXPLANATIONS - ⏳

- ++ CHANGE BAND VERBIAGE --> SEVERITY - ✅

- ++ RUBRIC SCORE VISUAL -> ADD TRANSPARENCY AND HORIZONTAL JITTER (A LITTLE BIT) - ✅

- ++ SPLIT SCATTER PLOT INTO 4 SECTIONS, USE THE CUTOFFS, (IE, ADD A VERTICAL LINE AT CUTOFF VALUE 1) - ⏳

- ++ ADDING A DISCREPENCY TO THE BAND (RENAME) CHART (RATHER THAN HOVER OVER, MAKE A VISUAL) - ⏳

