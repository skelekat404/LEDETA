#DESCRIPTION: audit.py defines the AuditLogger class, which creates structured JSONL audit records for reviewed cases. 
# Each record stores the timestamp, run ID, case identifiers, case window, number of emails, priority score, priority 
# band, and the explanation payload. This supports the dissertation’s governance, traceability, and auditability 
# requirements.

from __future__ import annotations #changes how Python handles type annotations so they are stored more efficiently. help modernize typing behavior
import json #allows for conversion to JSON
import os #file path compatability
from datetime import datetime #timestamps
from typing import Any, Dict, Optional ##Any = any Python type allowed, Dict = dict, Optional = a value may either be the specified type or None


class AuditLogger:
    def __init__(self, run_id: Optional[str] = None, path: str = "audit_log.jsonl") -> None:
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.path = os.path.abspath(path)

    def log_case(self, case_obj: Dict[str, Any], scored_row: Dict[str, Any], explanation: Dict[str, Any]) -> None:
        record = {
            "ts_utc": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "case_id": case_obj.get("case_id"),
            "employee": case_obj.get("employee"),
            "window_start": str(case_obj.get("window_start")),
            "window_end": str(case_obj.get("window_end")),
            "n_emails": int(case_obj.get("n_emails", 0)),
            "priority_score": float(scored_row.get("priority_score", 0.0)),
            "priority_band": str(scored_row.get("priority_band", "")),
            "explanation": explanation,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
