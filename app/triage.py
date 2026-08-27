"""Triage engine: negation-aware red-flag detection + urgency scoring.

Design principle (safety-first):
- NEGATED findings ("no pneumothorax") are NEVER escalated.
- UNCERTAIN findings ("cannot exclude PE") ARE escalated, but marked uncertain,
  because missing a possible emergency is worse than over-flagging.
- The final tier is the MAX over all affirmed/uncertain findings.
"""
from __future__ import annotations

import re

from .lexicon import (
    BENIGN,
    CRITICAL,
    ROUTINE,
    TIER_ADVICE,
    TIER_LABELS,
    TIER_ORDER,
    URGENT,
    RED_FLAGS_COMPILED,
)
from .negation import classify_assertion
from .sections import split_sections, split_sentences


def detect_findings(text: str) -> list[dict]:
    """Scan text for red-flag patterns with negation/uncertainty awareness.

    Returns a list of finding dicts:
      {pattern, tier, reason, matched_text, assertion, sentence, uncertain}
    Deduplicated by (tier, reason).
    """
    sections = split_sections(text)
    # Findings + impression carry the signal; scan everything but weight later.
    full_text = text

    findings: list[dict] = []
    # Dedupe by (tier, reason): if the same finding appears both affirmed and
    # uncertain, keep the AFFIRMED one (stronger evidence wins). We also
    # accumulate every matched text for a finding so downstream consumers
    # (eval harness, UI) see all evidence, not just the first regex hit.
    best: dict[tuple[str, str], dict] = {}
    matched_texts: dict[tuple[str, str], list[str]] = {}

    sentences = split_sentences(full_text)
    for sent in sentences:
        for pattern, tier, reason in RED_FLAGS_COMPILED:
            for m in pattern.finditer(sent):
                assertion = classify_assertion(sent, m.start(), m.end())
                if assertion == "negated":
                    continue  # safety: never escalate a negated finding
                key = (tier, reason)
                mt = m.group(0).strip()
                bucket = matched_texts.setdefault(key, [])
                if mt and mt not in bucket:
                    bucket.append(mt)
                entry = {
                    "tier": tier,
                    "reason": reason,
                    "matched_text": mt,
                    "assertion": assertion,
                    "uncertain": assertion == "uncertain",
                    "sentence": sent.strip(),
                }
                prev = best.get(key)
                if prev is None or (prev["uncertain"] and not entry["uncertain"]):
                    best[key] = entry

    findings = []
    for key, entry in best.items():
        entry["matched_text"] = " | ".join(matched_texts.get(key, []))
        findings.append(entry)
    return findings


def escalate_for_uncertainty(findings: list[dict]) -> list[dict]:
    """Uncertain critical findings stay critical (conservative). No-op kept
    explicit so the policy is auditable and testable."""
    return findings


def compute_triage(text: str) -> dict:
    """Return the triage verdict for a report.

    {
      tier, label, advice, findings:[...], counts:{critical,urgent,routine,benign},
      uncertain_flags:[...]
    }
    """
    findings = detect_findings(text)
    findings = escalate_for_uncertainty(findings)

    counts = {CRITICAL: 0, URGENT: 0, ROUTINE: 0, BENIGN: 0}
    for f in findings:
        counts[f["tier"]] = counts.get(f["tier"], 0) + 1

    if findings:
        tier = max((f["tier"] for f in findings), key=lambda t: TIER_ORDER[t])
        # Safety/clinical rule: an explicit "benign / negative for malignancy"
        # conclusion dominates over mere routine mentions (e.g. "nodule" in the
        # clinical history of a biopsy that came back benign). If the worst
        # finding is only routine but the report explicitly clears the patient,
        # report benign.
        has_explicit_benign = any(f["tier"] == BENIGN for f in findings)
        if has_explicit_benign and TIER_ORDER[tier] <= TIER_ORDER[ROUTINE]:
            tier = BENIGN
    else:
        # No red flags at all: check if the report reads as normal.
        lower = text.lower()
        if re.search(r"normal|unremarkable|no acute|no evidence of|negative for", lower):
            tier = BENIGN
        else:
            tier = ROUTINE

    # Sort findings by severity then certainty (affirmed before uncertain).
    findings.sort(
        key=lambda f: (-TIER_ORDER[f["tier"]], f["uncertain"], f["reason"])
    )

    return {
        "tier": tier,
        "label": TIER_LABELS[tier],
        "advice": TIER_ADVICE[tier],
        "findings": findings,
        "counts": {
            "critical": counts[CRITICAL],
            "urgent": counts[URGENT],
            "routine": counts[ROUTINE],
            "benign": counts[BENIGN],
        },
        "uncertain_flags": [f["reason"] for f in findings if f["uncertain"]],
    }
