from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Any, Optional

import pandas as pd


OPTIONAL_EMAIL_FIELDS = [
    "folder", "body_length", "subject_length",
    "from", "to", "cc", "bcc", "message_id", "file"
]


def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        # handle NaN
        if isinstance(x, float) and x != x:
            return ""
    except Exception:
        pass
    return str(x)


def build_cases(
    emails: pd.DataFrame,
    window_days: int = 30,
    body_mode: str = "excerpt",          # "excerpt" (default) or "full" or "none"
    excerpt_len: int = 800,              # only used for body_mode="excerpt"
) -> List[Dict[str, Any]]:
    """
    Case = all emails for one employee within NON-overlapping N-day windows.

    Performance goals:
    - Avoid storing huge full bodies inside cases by default (store excerpt instead).
    - Avoid pandas to_dict for each window (use itertuples -> faster).
    - Minimize copies and heavy object creation.

    Required columns: employee, date, subject, body (body can be missing if body_mode="none")
    """
    if emails is None or emails.empty:
        return []

    # Only keep the columns we need (reduces memory + speeds sorting)
    base_cols = ["employee", "date", "subject"]
    if "body" in emails.columns:
        base_cols.append("body")

    keep_cols = base_cols + [c for c in OPTIONAL_EMAIL_FIELDS if c in emails.columns]

    df = emails.loc[:, [c for c in keep_cols if c in emails.columns]].copy()

    # Parse dates once, drop null employee/date, sort once
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["employee", "date"])
    df = df.sort_values(["employee", "date"], kind="mergesort").reset_index(drop=True)

    # Stable row id for later lookup (optional future optimization)
    df["_row_id"] = df.index.astype(int)

    window_delta = timedelta(days=int(window_days))
    cases: List[Dict[str, Any]] = []

    # We will build email dicts with these fields.
    # NOTE: we always include "body" key if present, but it may be truncated/blank depending on mode.
    header_cols = ["_row_id", "date", "subject"] + [c for c in OPTIONAL_EMAIL_FIELDS if c in df.columns]
    has_body = "body" in df.columns

    # Group by employee (already sorted) — sort=False avoids extra sorting work
    for employee, grp in df.groupby("employee", sort=False):
        if grp.empty:
            continue

        # Convert dates to a list once for fast pointer scanning
        dates = grp["date"].tolist()
        n = len(grp)

        start = 0
        while start < n:
            window_start = dates[start]
            window_end = window_start + window_delta

            # Advance end pointer (non-overlapping windows)
            end = start
            while end < n and dates[end] < window_end:
                end += 1

            # Slice once
            window_slice = grp.iloc[start:end]

            # Build emails list FAST (avoid pandas to_dict per window)
            emails_out: List[Dict[str, Any]] = []

            # We build dicts from tuples; itertuples is much faster than to_dict
            # Use index=False so tuple order matches columns exactly
            tuple_cols = header_cols + (["body"] if has_body else [])
            for row in window_slice.loc[:, tuple_cols].itertuples(index=False, name=None):
                # Map tuple -> dict without pandas overhead
                # Build header dict first
                d: Dict[str, Any] = {}
                # header fields
                for i, col in enumerate(header_cols):
                    d[col if col != "_row_id" else "email_row_id"] = row[i]

                # body handling
                if has_body:
                    body_val = row[len(header_cols)]
                    if body_mode == "none":
                        d["body"] = ""
                    elif body_mode == "full":
                        d["body"] = _safe_str(body_val)
                    else:
                        # excerpt default
                        s = _safe_str(body_val)
                        d["body"] = s[: int(excerpt_len)] if excerpt_len and excerpt_len > 0 else s

                emails_out.append(d)

            case_id = f"{employee}::{window_start.date()}::{window_end.date()}"

            cases.append({
                "case_id": case_id,
                "employee": employee,
                "window_start": window_start,
                "window_end": window_end,
                "emails": emails_out,
                "n_emails": int(len(emails_out)),
            })

            start = end

    return cases
