from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any

from ledeta.features import extract_engineered_features, DEFAULT_KEYWORDS


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _log_norm(x: float, ref: float) -> float:
    x = max(0.0, float(x))
    ref = max(1.0, float(ref))
    return math.log1p(x) / math.log1p(ref)


def _saturate_to_0_100(raw: float, denom: float = 28.0) -> float:
    """
    Map a raw positive score to 0..100 smoothly.
    Lower denom => reaches 100 faster. For ethics triage, we want MORE separation at midrange,
    and fewer "Critical" by default; the gating logic below handles Critical rarity.
    """
    raw = max(0.0, float(raw))
    score = 100.0 * (1.0 - math.exp(-raw / float(denom)))
    return float(_clamp(score, 0.0, 100.0))


def score_case_rubric_v3(case: Dict, keywords: List[str] = DEFAULT_KEYWORDS) -> Dict[str, Any]:
    """
    Ethics-focused scoring (Enron-friendly):
      - prioritizes off-channel/secrecy, approval/control bypass, record cleanup/tampering,
        urgency + low specificity, distancing language (if present), and mild money-ops signals.
      - applies a marketing/spam penalty to triage score
      - HARD SPAM FILTER flag to drop obvious newsletters/promo noise

    Returns a dict with:
      - fraud_score (0..100): (kept for backwards compatibility) now means "ethics_score"
      - triage_score (0..100): ethics_score minus spam penalty (clamped and gated)
      - spam_penalty (0..45)
      - spam_filtered (bool): True if case is very likely newsletter/promo AND low ethics risk
      - reasons: list[str]
      - components: dict of point components (pre-saturation)
    """
    feats = extract_engineered_features(case, keywords=keywords)
    reasons: List[str] = []

    n_emails = float(feats.get("n_emails", case.get("n_emails", 0)) or 0.0)
    n_emails = max(0.0, n_emails)

    # -----------------------------
    # 0) Spam diagnostics (for penalty + filtering)
    # -----------------------------
    spam_signal = float(feats.get("spam_signal", 0.0) or 0.0)
    has_unsubscribe = float(feats.get("has_unsubscribe", 0.0) or 0.0)
    url_count = float(feats.get("url_count", 0.0) or 0.0)
    marketing_hits = float(feats.get("marketing_hits", 0.0) or 0.0)

    spam_norm = _clamp(spam_signal / 8.0, 0.0, 1.0)
    spam_penalty = 35.0 * spam_norm
    if has_unsubscribe >= 1.0 and url_count >= 1.0:
        spam_penalty += 10.0
    spam_penalty = float(_clamp(spam_penalty, 0.0, 45.0))

    # -----------------------------
    # 1) Unethical / policy-evading behavior block (dominant)
    # -----------------------------
    offline_secrecy_hits = float(feats.get("offline_secrecy_hits", 0.0) or 0.0)
    approval_bypass_hits = float(feats.get("approval_bypass_hits", 0.0) or 0.0)
    record_tamper_hits = float(feats.get("record_tamper_hits", 0.0) or 0.0)
    vague_expense_hits = float(feats.get("vague_expense_hits", 0.0) or 0.0)
    low_specificity_flag = float(feats.get("low_specificity_flag", 0.0) or 0.0)
    urgent_subjects = float(feats.get("urgent_subjects", 0.0) or 0.0)

    # Optional (only if your features.py computes these)
    distancing_hits = float(feats.get("distancing_hits", 0.0) or 0.0)

    # Mild money-ops support (fits scope but not dominant)
    money_ops_hits = float(feats.get("money_ops_hits", 0.0) or 0.0)
    bank_change_hits = float(feats.get("bank_change_hits", 0.0) or 0.0)
    mentions_bank_account = float(feats.get("mentions_bank_account", 0.0) or 0.0)

    # Use rate-based versions where it makes sense (prevents big cases from dominating)
    urgent_rate = urgent_subjects / max(1.0, n_emails)
    urgent_norm = _clamp(urgent_rate / 0.15, 0.0, 1.0)  # slightly harder than before

    # Ethics raw points: make "concealment + bypass + cleanup" matter most
    ethics_core_raw = (
        3.2 * offline_secrecy_hits
        + 2.8 * approval_bypass_hits
        + 2.6 * record_tamper_hits
        + 1.4 * vague_expense_hits
        + 1.2 * low_specificity_flag
        + 1.3 * distancing_hits
        + 1.8 * urgent_norm  # bounded
        # Mild financial-ish support (still within unethical scope)
        + 0.6 * money_ops_hits
        + 0.7 * bank_change_hits
        + 0.7 * mentions_bank_account
    )

    # Normalize ethics core into a bounded contribution
    # Higher divisor makes it harder to max out -> fewer "Critical"
    ethics_norm = _clamp(ethics_core_raw / 14.0, 0.0, 1.0)
    ethics_pts = 70.0 * ethics_norm  # dominant block

    # -----------------------------
    # 2) Context (small, rate-limited)
    # -----------------------------
    volume_norm = _log_norm(n_emails, ref=300.0)
    volume_pts = 6.0 * _clamp(volume_norm, 0.0, 1.0)

    max_day = float(feats.get("max_emails_in_a_day", 0.0) or 0.0)
    burst_norm = _log_norm(max_day, ref=60.0)
    burst_pts = 6.0 * _clamp(burst_norm, 0.0, 1.0)

    # Raw ethics score (no spam penalty)
    ethics_raw = ethics_pts + volume_pts + burst_pts

    # -----------------------------
    # 3) Hard spam filter flag (drop obvious newsletters)
    # -----------------------------
    # If it is very spammy AND ethics signal is low, treat as spam-only and filter out
    # Thresholds tuned to be conservative (avoid filtering real misconduct accidentally).
    spam_filtered = bool(
        (spam_signal >= 6.0 or (has_unsubscribe >= 1.0 and url_count >= 2.0) or marketing_hits >= 3.0)
        and ethics_core_raw < 3.5
    )

    # -----------------------------
    # 4) Triage score = ethics minus spam penalty (then gated to reduce "Critical spam")
    # -----------------------------
    triage_raw = max(0.0, ethics_raw - spam_penalty)

    ethics_score = _saturate_to_0_100(ethics_raw, denom=28.0)
    triage_score = _saturate_to_0_100(triage_raw, denom=28.0)

    # Critical rarity gate:
    # Require at least TWO of the three strongest “misconduct categories” before allowing > 75.
    category_hits = 0
    if offline_secrecy_hits > 0:
        category_hits += 1
    if approval_bypass_hits > 0:
        category_hits += 1
    if record_tamper_hits > 0:
        category_hits += 1

    if category_hits < 2 and triage_score > 75.0:
        triage_score = 75.0  # cap at "High" unless multi-category misconduct

    # -----------------------------
    # 5) Reasons (clear wording for your new scope)
    # -----------------------------
    reasons.append(
        f"Ethics score {ethics_score:.1f}/100 emphasizes off-channel/secrecy, bypassing controls, and record cleanup behaviors."
    )
    if spam_penalty > 0:
        reasons.append(f"Triage applies a marketing/spam penalty −{spam_penalty:.1f} to reduce newsletters/promos.")
    if spam_filtered:
        reasons.append("Case flagged as spam/newsletter-dominant and will be filtered from the queue by default.")
    reasons.append(f"Triage score {triage_score:.1f}/100 is used for queue ranking.")

    # Optional detail reasons
    if offline_secrecy_hits > 0:
        reasons.append("Off-channel / secrecy language present.")
    if approval_bypass_hits > 0:
        reasons.append("Approval/control bypass language present.")
    if record_tamper_hits > 0:
        reasons.append("Record cleanup/tampering language present.")
    if low_specificity_flag > 0 and urgent_subjects > 0:
        reasons.append("Urgency + low specificity pattern present.")

    components = {
        "ethics_pts": ethics_pts,
        "volume_pts": volume_pts,
        "burst_pts": burst_pts,
        "ethics_raw": ethics_raw,
        "spam_penalty": spam_penalty,
        "triage_raw": triage_raw,
        "spam_filtered": spam_filtered,
        "ethics_core_raw": ethics_core_raw,
        "category_hits": category_hits,
    }

    # NOTE: Keep legacy keys for the rest of your app
    return {
        "fraud_score": ethics_score,      # legacy key (now ethics_score)
        "triage_score": triage_score,
        "spam_penalty": spam_penalty,
        "spam_filtered": spam_filtered,
        "reasons": reasons,
        "components": components,
        # nice-to-have aliases (won’t break anything)
        "ethics_score": ethics_score,
    }


def score_case_rubric(case: Dict, keywords: List[str] = DEFAULT_KEYWORDS) -> Tuple[float, List[str]]:
    """
    Backward-compatible function used throughout the app/model.
    Returns triage_score as the primary "priority_score".
    """
    res = score_case_rubric_v3(case, keywords=keywords)
    return float(res["triage_score"]), list(res["reasons"])
