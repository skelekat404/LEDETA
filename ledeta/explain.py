from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple
import pandas as pd

from ledeta.features import (
    extract_engineered_features,
    DEFAULT_KEYWORDS,
    MONEY_OPS_TERMS,
    BANK_CHANGE_TERMS,
    APPROVAL_BYPASS_TERMS,
    RECORD_TAMPER_TERMS,
    SECRECY_OFFLINE_TERMS,
    VAGUE_EXPENSE_TERMS,
    DISTANCING_TERMS,
    MARKETING_TERMS,
)


def _safe_text(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and x != x:
            return ""
    except Exception:
        pass
    return str(x)


def _safe_num(x, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if isinstance(x, float) and x != x:
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def _normalize_emails(case: Dict) -> List[Dict[str, Any]]:
    emails = case.get("emails", [])
    if emails is None:
        return []
    if isinstance(emails, list):
        return emails
    return []


def _make_snip(e: Dict[str, Any], excerpt_len: int = 420) -> Dict[str, str]:
    body = _safe_text(e.get("body"))
    excerpt = body[:excerpt_len]
    return {
        "date": _safe_text(e.get("date", "")),
        "subject": _safe_text(e.get("subject", "(no subject)")) or "(no subject)",
        "excerpt": excerpt,
        "folder": _safe_text(e.get("folder", "")),
        "from": _safe_text(e.get("from", "")),
        "to": _safe_text(e.get("to", "")),
    }


def _fallback_snippets(case: Dict, k: int = 6) -> List[Dict[str, str]]:
    emails = _normalize_emails(case)
    if not emails:
        return []

    rows = []
    for i, e in enumerate(emails):
        rows.append({
            "i": i,
            "date": e.get("date"),
            "body_len": len(_safe_text(e.get("body"))),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].notna().any():
        df = df.sort_values("date", ascending=False)
    else:
        df = df.sort_values("body_len", ascending=False)

    take_idx = df["i"].head(k).tolist()
    return [_make_snip(emails[i]) for i in take_idx]


def _risk_tags_from_features(feats: Dict[str, float]) -> List[str]:
    tags: List[str] = []

    spam_signal = _safe_num(feats.get("spam_signal"), 0.0)
    if spam_signal >= 6.0 or _safe_num(feats.get("has_unsubscribe"), 0.0) >= 1.0:
        tags.append("Likely marketing/spam")

    if _safe_num(feats.get("offline_secrecy_hits"), 0.0) > 0:
        tags.append("Off-channel / secrecy risk")

    if _safe_num(feats.get("approval_bypass_hits"), 0.0) > 0:
        tags.append("Control/approval bypass risk")

    if _safe_num(feats.get("record_tamper_hits"), 0.0) > 0:
        tags.append("Record cleanup/tampering risk")

    if _safe_num(feats.get("vague_expense_hits"), 0.0) > 0:
        tags.append("Vague request/vendor risk")

    if _safe_num(feats.get("low_specificity_flag"), 0.0) > 0 and _safe_num(feats.get("urgent_subjects"), 0.0) > 0:
        tags.append("Urgency + low specificity pattern")

    # Mild money ops tags (fits scope but not dominant)
    if _safe_num(feats.get("bank_change_hits"), 0.0) > 0 or _safe_num(feats.get("mentions_bank_account"), 0.0) > 0:
        tags.append("Bank/wire-change language present")

    return tags[:6]


def _term_groups(feats: Dict[str, float]) -> List[str]:
    """
    Choose which term families to prioritize based on features.
    This prevents scanning for everything when most cases are low-risk.
    """
    groups: List[str] = []

    if _safe_num(feats.get("offline_secrecy_hits"), 0.0) > 0:
        groups.append("secrecy")
    if _safe_num(feats.get("approval_bypass_hits"), 0.0) > 0:
        groups.append("bypass")
    if _safe_num(feats.get("record_tamper_hits"), 0.0) > 0:
        groups.append("tamper")
    if _safe_num(feats.get("vague_expense_hits"), 0.0) > 0:
        groups.append("vague")
    if _safe_num(feats.get("distancing_hits"), 0.0) > 0:
        groups.append("distancing")

    # include money ops lightly if present
    if _safe_num(feats.get("money_ops_hits"), 0.0) > 0 or _safe_num(feats.get("bank_change_hits"), 0.0) > 0:
        groups.append("money")

    # spam terms only if spammy
    if _safe_num(feats.get("spam_signal"), 0.0) >= 6.0:
        groups.append("marketing")

    # always include a small set of keywords
    groups.append("keywords")

    # keep order & uniqueness
    seen = set()
    out = []
    for g in groups:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _terms_for_groups(groups: List[str]) -> List[str]:
    terms: List[str] = []
    for g in groups:
        if g == "secrecy":
            terms += SECRECY_OFFLINE_TERMS
        elif g == "bypass":
            terms += APPROVAL_BYPASS_TERMS
        elif g == "tamper":
            terms += RECORD_TAMPER_TERMS
        elif g == "vague":
            terms += VAGUE_EXPENSE_TERMS
        elif g == "distancing":
            terms += DISTANCING_TERMS
        elif g == "money":
            terms += (BANK_CHANGE_TERMS + MONEY_OPS_TERMS)
        elif g == "marketing":
            terms += MARKETING_TERMS
        elif g == "keywords":
            terms += DEFAULT_KEYWORDS
    # normalize + de-dupe preserving order
    seen: Set[str] = set()
    cleaned: List[str] = []
    for t in terms:
        tt = (t or "").strip()
        if not tt:
            continue
        low = tt.lower()
        if low not in seen:
            seen.add(low)
            cleaned.append(tt)
    return cleaned


def _collect_salient_terms_and_snips(case: Dict, terms: List[str], k_snips: int = 8) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    FAST path:
    - scan email-by-email
    - stop when we have enough snippets
    - avoid building one giant joined string
    """
    emails = _normalize_emails(case)
    if not emails or not terms:
        return [], []

    lowered_terms = [(t, t.lower()) for t in terms if t]
    found_terms: List[str] = []
    found_terms_set: Set[str] = set()
    snippets: List[Dict[str, str]] = []

    for e in emails:
        subj = _safe_text(e.get("subject", ""))
        body = _safe_text(e.get("body", ""))

        # limit scan body length to keep it fast (still enough for signals)
        text = (subj + " " + body[:2000]).lower()

        hit_any = False
        for orig, low in lowered_terms:
            if low in text:
                if low not in found_terms_set:
                    found_terms_set.add(low)
                    found_terms.append(orig)
                hit_any = True

        if hit_any:
            snippets.append(_make_snip(e))
            if len(snippets) >= k_snips:
                break

        # stop early if we already have enough salient terms and some snippets
        if len(found_terms) >= 12 and len(snippets) >= 4:
            break

    return found_terms[:12], snippets


def explain_case(case: Dict, scored_row: Dict[str, Any]) -> Dict[str, Any]:
    score = _safe_num(scored_row.get("priority_score"), default=0.0)
    feats = extract_engineered_features(case)

    drivers = [
        {"name": "offline_secrecy_hits", "value": feats.get("offline_secrecy_hits"), "note": "Off-channel/secrecy language (call me, don’t email, offline)"},
        {"name": "approval_bypass_hits", "value": feats.get("approval_bypass_hits"), "note": "Approval/control bypass language"},
        {"name": "record_tamper_hits", "value": feats.get("record_tamper_hits"), "note": "Record cleanup/tampering language"},
        {"name": "low_specificity_flag", "value": feats.get("low_specificity_flag"), "note": "Low specificity / vague request flag"},
        {"name": "urgent_subjects", "value": feats.get("urgent_subjects"), "note": "Urgency terms in subject"},
        {"name": "distancing_hits", "value": feats.get("distancing_hits", 0.0), "note": "Distancing/evasive language (if enabled)"},
        {"name": "spam_signal", "value": feats.get("spam_signal"), "note": "Marketing/spam composite signal"},
        {"name": "marketing_hits", "value": feats.get("marketing_hits", 0.0), "note": "Marketing terms (newsletter/promo/unsubscribe)"},
        {"name": "url_count", "value": feats.get("url_count"), "note": "Number of URLs"},
        {"name": "n_emails", "value": feats.get("n_emails"), "note": "Case communication volume"},
        {"name": "max_emails_in_a_day", "value": feats.get("max_emails_in_a_day"), "note": "Burst indicator"},
        # mild money ops (fits scope but not dominant)
        {"name": "money_ops_hits", "value": feats.get("money_ops_hits", 0.0), "note": "Money-operations terms (invoice/payment/vendor)"},
        {"name": "bank_change_hits", "value": feats.get("bank_change_hits", 0.0), "note": "Bank/wire-change language"},
    ]

    for d in drivers:
        d["value"] = _safe_num(d.get("value"), default=0.0)

    top_features = sorted(drivers, key=lambda d: d["value"], reverse=True)[:8]
    risk_tags = _risk_tags_from_features(feats)

    # Choose term groups based on features (faster + more relevant)
    groups = _term_groups(feats)
    terms = _terms_for_groups(groups)

    salient_terms, snippets = _collect_salient_terms_and_snips(case, terms=terms, k_snips=8)
    if not snippets:
        snippets = _fallback_snippets(case, k=6)

    summary = (
        f"Priority score **{score:.1f}/100** (ethics/unethical-behavior triage). "
        f"Off-channel={_safe_num(feats.get('offline_secrecy_hits')):.1f}, "
        f"Bypass={_safe_num(feats.get('approval_bypass_hits')):.1f}, "
        f"Cleanup={_safe_num(feats.get('record_tamper_hits')):.1f}, "
        f"Spam={_safe_num(feats.get('spam_signal')):.1f}. "
        f"Tags: {', '.join(risk_tags) if risk_tags else 'None'}."
    )

    return {
        "summary": summary,
        "risk_tags": risk_tags,
        "top_features": top_features,
        "salient_terms": salient_terms,
        "snippets": snippets,
        "engineered_features": feats,
        "explain_version": "ethics_mvp_v2.0",
    }
