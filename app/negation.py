"""Negation & uncertainty detection (ConNeg-style, tuned for radiology/pathology).

Given a sentence and a matched finding span, decide whether the finding is
AFFIRMED, NEGATED, or UNCERTAIN. This is the single most important safety
component: "no pneumothorax" must never be flagged as a pneumothorax.
"""
from __future__ import annotations

import re

from .lexicon import (
    POST_NEGATION_COMPILED,
    PRE_NEGATION_COMPILED,
    UNCERTAINTY_COMPILED,
)

# Window (chars) around a finding span to search for cues.
PRE_WINDOW = 60
POST_WINDOW = 40

# "cannot exclude / cannot rule out" is UNCERTAINTY, not negation. This must
# be checked BEFORE negation cues, because "exclude"/"rule out" also appear in
# the negation list and would otherwise wrongly suppress real emergencies.
_CANNOT_EXCLUDE_RE = re.compile(
    r"(?:cannot|can'?t|could\s+not|unable\s+to|not\s+(?:yet\s+)?)\s+"
    r"(?:exclude|rule\s+out|excluded|ruled\s+out|be\s+excluded)",
    re.IGNORECASE,
)


def classify_assertion(sentence: str, span_start: int, span_end: int) -> str:
    """Return 'affirmed' | 'negated' | 'uncertain' for a finding span.

    Order of precedence: negation > uncertainty > affirmed. A negation cue
    anywhere in the pre-window (or a post-negation cue right after the span)
    wins, because "no evidence of X" is unambiguous. Uncertainty cues
    ("cannot exclude X", "suspicious for X") keep the finding but mark it
    uncertain so triage can still act on it conservatively.
    """
    pre = sentence[max(0, span_start - PRE_WINDOW):span_start]
    post = sentence[span_end:span_end + POST_WINDOW]

    # --- 0. "Cannot exclude / cannot rule out" -> uncertain (safety-first) ---
    if _CANNOT_EXCLUDE_RE.search(pre) or _CANNOT_EXCLUDE_RE.search(post):
        return "uncertain"

    # --- Negation ---
    for pat in PRE_NEGATION_COMPILED:
        m = pat.search(pre)
        if m:
            # Cue must be close to the finding (no intervening period).
            tail = pre[m.end():]
            if "." not in tail:
                return "negated"
    for pat in POST_NEGATION_COMPILED:
        m = pat.match(post)
        if m:
            return "negated"

    # --- Uncertainty ---
    for pat in UNCERTAINTY_COMPILED:
        m = pat.search(pre)
        if m and "." not in pre[m.end():]:
            return "uncertain"
    for pat in UNCERTAINTY_COMPILED:
        if pat.match(post):
            return "uncertain"

    return "affirmed"


def find_matches_with_assertion(
    sentence: str, pattern: re.Pattern
) -> list[dict]:
    """Find all regex matches in a sentence and classify each assertion."""
    out = []
    for m in pattern.finditer(sentence):
        assertion = classify_assertion(sentence, m.start(), m.end())
        out.append(
            {
                "text": m.group(0),
                "start": m.start(),
                "end": m.end(),
                "assertion": assertion,
            }
        )
    return out
