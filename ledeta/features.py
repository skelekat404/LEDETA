from __future__ import annotations
from typing import Dict, List, Optional
import re
import math


def _safe_text(x) -> str:
    try:
        if x is None:
            return ""
        return str(x)
    except Exception:
        return ""


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


# -----------------------------
# Keyword groups (fraud + spam)
# -----------------------------
# You can expand these over time. Keep them defensible: "process/control bypass" > "embezzle".
MONEY_OPS_TERMS = [
    "invoice", "payment", "wire", "transfer", "ach", "check", "refund", "reimbursement",
    "vendor", "supplier", "purchase order", "po ", "p.o.", "billing", "bill", "remit",
    "bank", "routing", "account number", "swift", "iban", "settlement", "balance", "ledger"
]

BANK_CHANGE_TERMS = [
    "updated banking", "change bank", "change banking", "new bank",
    "update routing", "update account", "new account", "new routing",
    "wire instructions", "remit to", "different account", "alternate account"
]

APPROVAL_BYPASS_TERMS = [
    "skip approval", "no approval", "avoid approval", "no need to document", "don't document",
    "off the books", "not on the books", "bypass", "override", "exception",
    "do not involve", "keep it between us", "don't forward", "confidential"
]

RECORD_TAMPER_TERMS = [
    "backdate", "back date", "reclass", "re-class", "reclassify", "adjustment", "journal entry",
    "move it", "move this", "clean up", "cleanup", "remove", "delete", "wipe", "destroy"
]

SECRECY_OFFLINE_TERMS = [
    "call me", "let's talk", "offline", "in person", "on my cell", "text me",
    "per our conversation", "as discussed", "verbal", "do not email"
]

VAGUE_EXPENSE_TERMS = [
    "misc", "miscellaneous", "services rendered", "consulting", "advisory",
    "facilitation", "special arrangement", "side agreement", "handling fee"
]

DISTANCING_TERMS = [
    "i was told", "i was instructed", "not sure", "i don't recall",
    "someone said", "we were told", "per instructions", "per direction"
]

# Marketing/spam indicators (downrankers)
MARKETING_TERMS = [
    "unsubscribe", "subscribe", "newsletter", "promotion", "promo", "offer",
    "sale", "discount", "deal", "limited time", "click here", "view online",
    "free", "win", "winner", "congratulations", "register now", "marketing"
]


# -----------------------------
# Default keywords (kept for backwards compatibility)
# -----------------------------
# This is now a *blend* but you'll mostly use the grouped features below.
DEFAULT_KEYWORDS = [
    "wire", "transfer", "invoice", "payment", "urgent", "confidential",
    "kickback", "bribe", "offshore", "shell", "launder", "fraud", "embezzle",
]


# -----------------------------
# Regex patterns
# -----------------------------
_money_re = re.compile(r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?")
_caps_re = re.compile(r"\b[A-Z]{4,}\b")
_email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_url_re = re.compile(r"https?://\S+|www\.\S+")
_invoice_id_re = re.compile(r"\b(inv|invoice)\s*#?\s*([a-z0-9\-]{3,})\b", re.IGNORECASE)
_po_re = re.compile(r"\b(po|p\.o\.)\s*#?\s*([a-z0-9\-]{3,})\b", re.IGNORECASE)
_account_re = re.compile(r"\b(account|acct)\b", re.IGNORECASE)
_routing_re = re.compile(r"\brouting\b", re.IGNORECASE)


FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "proton.me", "protonmail.com"
}


def _recipient_count(to_field) -> int:
    if to_field is None:
        return 0
    if isinstance(to_field, float):
        return 0
    txt = str(to_field)
    return len(_email_re.findall(txt))


def _count_term_hits(text: str, terms: List[str]) -> int:
    """
    Count occurrences with simple substring logic (defensible, fast).
    Uses lowercase text; terms should be lowercase-ish.
    """
    t = text.lower()
    hits = 0
    for term in terms:
        if not term:
            continue
        hits += t.count(term.lower())
    return hits


def _count_term_docs(text: str, terms: List[str]) -> int:
    """Count how many unique terms appear at least once."""
    t = text.lower()
    return sum(1 for term in terms if term and term.lower() in t)


def _extract_domains(s: str) -> List[str]:
    s = _safe_text(s)
    return [m.split("@")[-1].lower() for m in _email_re.findall(s)]


def extract_engineered_features(case: Dict, keywords: Optional[List[str]] = None) -> Dict[str, float]:
    kw = keywords or DEFAULT_KEYWORDS
    emails = case.get("emails", []) or []

    # Aggregate full text for scans
    text_all = " ".join(
        [f"{_safe_text(e.get('subject'))} {_safe_text(e.get('body'))}" for e in emails]
    ).lower()

    feats: Dict[str, float] = {}
    feats["n_emails"] = float(case.get("n_emails", len(emails)))

    # -----------------------------
    # Legacy keyword indicators
    # -----------------------------
    feats["keyword_hits"] = float(_count_term_hits(text_all, kw))
    feats["unique_keywords_hit"] = float(_count_term_docs(text_all, kw))

    # -----------------------------
    # Fraud-related grouped signals
    # -----------------------------
    feats["money_ops_hits"] = float(_count_term_hits(text_all, MONEY_OPS_TERMS))
    feats["money_ops_unique"] = float(_count_term_docs(text_all, MONEY_OPS_TERMS))

    feats["bank_change_hits"] = float(_count_term_hits(text_all, BANK_CHANGE_TERMS))
    feats["approval_bypass_hits"] = float(_count_term_hits(text_all, APPROVAL_BYPASS_TERMS))
    feats["record_tamper_hits"] = float(_count_term_hits(text_all, RECORD_TAMPER_TERMS))
    feats["offline_secrecy_hits"] = float(_count_term_hits(text_all, SECRECY_OFFLINE_TERMS))
    feats["vague_expense_hits"] = float(_count_term_hits(text_all, VAGUE_EXPENSE_TERMS))
    feats["distancing_hits"] = float(_count_term_hits(text_all, DISTANCING_TERMS))

    # Presence flags for higher-signal patterns
    feats["has_invoice_id"] = float(1.0 if _invoice_id_re.search(text_all) else 0.0)
    feats["has_po_number"] = float(1.0 if _po_re.search(text_all) else 0.0)
    feats["mentions_bank_account"] = float(1.0 if (_account_re.search(text_all) and _routing_re.search(text_all)) else 0.0)

    # -----------------------------
    # Marketing / spam indicators (DOWNRANKERS)
    # -----------------------------
    feats["marketing_hits"] = float(_count_term_hits(text_all, MARKETING_TERMS))
    feats["has_unsubscribe"] = float(1.0 if "unsubscribe" in text_all else 0.0)
    feats["url_count"] = float(len(_url_re.findall(text_all)))

    # Subject-heavy marketing clues
    subj_all = " ".join([_safe_text(e.get("subject")) for e in emails]).lower()
    feats["marketing_subject_hits"] = float(_count_term_hits(subj_all, MARKETING_TERMS))

    # -----------------------------
    # Money / urgency / formatting
    # -----------------------------
    feats["money_mentions"] = float(len(_money_re.findall(text_all)))

    feats["urgent_subjects"] = float(
        sum(1 for e in emails if "urgent" in _safe_text(e.get("subject")).lower())
    )

    feats["all_caps_tokens"] = float(len(_caps_re.findall(" ".join([_safe_text(e.get("subject")) for e in emails]))))

    # -----------------------------
    # Temporal burst: max emails per day
    # -----------------------------
    by_day = {}
    for e in emails:
        d = e.get("date")
        if hasattr(d, "date"):
            dd = d.date()
        else:
            dd = None
        if dd is None:
            continue
        by_day[dd] = by_day.get(dd, 0) + 1
    feats["max_emails_in_a_day"] = float(max(by_day.values()) if by_day else 0.0)

    # -----------------------------
    # Length / specificity
    # -----------------------------
    body_lengths = []
    subj_lengths = []
    token_counts = []

    for e in emails:
        body = _safe_text(e.get("body"))
        subj = _safe_text(e.get("subject"))

        if "body_length" in e and e["body_length"] is not None:
            body_lengths.append(_safe_float(e["body_length"], default=0.0))
        else:
            body_lengths.append(float(len(body)))

        if "subject_length" in e and e["subject_length"] is not None:
            subj_lengths.append(_safe_float(e["subject_length"], default=0.0))
        else:
            subj_lengths.append(float(len(subj)))

        # crude specificity proxy: word count
        token_counts.append(float(len((subj + " " + body).split())))

    feats["mean_body_length"] = float(sum(body_lengths) / len(body_lengths)) if body_lengths else 0.0
    feats["max_body_length"] = float(max(body_lengths)) if body_lengths else 0.0
    feats["mean_subject_length"] = float(sum(subj_lengths) / len(subj_lengths)) if subj_lengths else 0.0
    feats["mean_token_count"] = float(sum(token_counts) / len(token_counts)) if token_counts else 0.0

    # “Low specificity” heuristic: short messages but high urgency/money ops
    feats["low_specificity_flag"] = float(
        1.0 if (feats["mean_token_count"] < 35 and (feats["urgent_subjects"] > 0 or feats["money_ops_hits"] > 0)) else 0.0
    )

    # -----------------------------
    # Communication breadth (recipients)
    # -----------------------------
    to_counts = [_recipient_count(e.get("to")) for e in emails]
    feats["mean_recipients_to"] = float(sum(to_counts) / len(to_counts)) if to_counts else 0.0
    feats["max_recipients_to"] = float(max(to_counts)) if to_counts else 0.0

    # -----------------------------
    # External contact heuristics
    # -----------------------------
    from_domains = []
    to_domains = []

    for e in emails:
        from_domains += _extract_domains(e.get("from"))
        to_domains += _extract_domains(e.get("to"))

    # ratio of emails involving free email domains (often spammy or risky)
    free_dom_hits = sum(1 for d in (from_domains + to_domains) if d in FREE_EMAIL_DOMAINS)
    feats["free_email_domain_hits"] = float(free_dom_hits)

    # Basic external-ness proxy: if domains exist and many are not "enron.com"
    # (If your dataset uses a different corporate domain, adjust here.)
    noncorp = 0
    corp = 0
    for d in (from_domains + to_domains):
        if not d:
            continue
        if d.endswith("enron.com"):
            corp += 1
        else:
            noncorp += 1
    feats["noncorp_domain_hits"] = float(noncorp)
    feats["corp_domain_hits"] = float(corp)
    feats["noncorp_domain_ratio"] = float(noncorp / (noncorp + corp)) if (noncorp + corp) > 0 else 0.0

    # -----------------------------
    # Folder presence (neutral)
    # -----------------------------
    feats["has_folder_field"] = float(
        1.0 if any(("folder" in e and e.get("folder") not in (None, "")) for e in emails) else 0.0
    )

    # -----------------------------
    # Composite signals (these are the levers you’ll later use to downrank spam)
    # -----------------------------
    # Fraud signal: prioritize control bypass + bank changes + money ops + tampering
    fraud_signal = (
        2.5 * feats["bank_change_hits"]
        + 2.0 * feats["approval_bypass_hits"]
        + 1.8 * feats["record_tamper_hits"]
        + 1.2 * feats["money_ops_hits"]
        + 1.0 * feats["offline_secrecy_hits"]
        + 0.8 * feats["vague_expense_hits"]
        + 0.6 * feats["distancing_hits"]
        + 1.0 * feats["mentions_bank_account"]
        + 0.6 * feats["low_specificity_flag"]
    )

    # Spam signal: unsubscribe + links + marketing language
    spam_signal = (
        4.0 * feats["has_unsubscribe"]
        + 1.0 * feats["marketing_hits"]
        + 1.5 * feats["marketing_subject_hits"]
        + 0.7 * feats["url_count"]
    )

    feats["fraud_signal"] = float(fraud_signal)
    feats["spam_signal"] = float(spam_signal)

    # “Net risk” is helpful later if you want a single score that downranks spam
    feats["net_risk_signal"] = float(max(0.0, fraud_signal - 1.25 * spam_signal))

    return feats


def case_to_document(case: Dict) -> str:
    emails = case.get("emails", []) or []
    parts = []
    for e in emails:
        parts.append((_safe_text(e.get("subject")) + " " + _safe_text(e.get("body"))).strip())
    return "\n".join(parts)
