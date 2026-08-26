"""Deterministic extractive summarizer for radiology/pathology reports.

Produces:
- plain_summary: patient-friendly bullet summary (extractive, grounded)
- key_findings: the most important sentences, ranked
- measurements: extracted sizes/percentages with context
- glossary_terms: jargon found in the report with plain definitions
- sections: parsed report sections

Everything here is extractive — no invented content — which is what makes it
safe to run without an LLM and what the faithfulness gate checks against.
"""
from __future__ import annotations

import re

from .lexicon import GLOSSARY, MEASUREMENT_CONTEXT_RE, MEASUREMENT_RE
from .sections import split_sections, split_sentences
from .triage import compute_triage

# Sentences that are pure boilerplate (not worth surfacing).
_BOILERPLATE_RE = re.compile(
    r"^(comparison|technique|exam|examination|study|method|procedure|indication|"
    r"clinical history|history|reason for|prior|previous|contrast|dose|"
    r"the (?:images|scans?|films?|study) (?:were|was) reviewed|"
    r"this (?:exam|study|report)|i (?:have|was)|signed|electronically)",
    re.IGNORECASE,
)

# Words that make a sentence "informative" for ranking.
_INFORMATIVE_RE = re.compile(
    r"(?:show|shows|demonstrate|reveals?|identified|seen|noted|present|"
    r"measur|enlarged|mass|nodule|lesion|fracture|effusion|consolidation|"
    r"stenosis|occlusion|thrombus|thrombosis|embol|hemorrhage|hematoma|"
    r"infarct|edema|abscess|pneumonia|tumor|carcinoma|malignant|benign|"
    r"biopsy|cells|diagnosis|impression|conclusion|stable|unchanged|"
    r"interval|new|increased|decreased|improved|worsened|normal|"
    r"unremarkable|negative|positive|no evidence|free air|calcif|cyst|"
    r"stone|calculus|obstruc|dilated|patent|compress)",
    re.IGNORECASE,
)


def _rank_sentence(sent: str, idx: int, total: int) -> float:
    score = 0.0
    if _INFORMATIVE_RE.search(sent):
        score += 3.0
    if MEASUREMENT_RE.search(sent):
        score += 1.5
    # Impression/conclusion sentences are usually the headline.
    if re.search(r"impression|conclusion|diagnosis", sent, re.IGNORECASE):
        score += 2.0
    # Mild recency boost (findings often summarized late), but keep order stable.
    if total > 1:
        score += 0.5 * (idx / (total - 1))
    # Penalize very short fragments.
    if len(sent) < 25:
        score -= 1.0
    return score


def extract_measurements(text: str) -> list[dict]:
    """Extract measurements with surrounding context (what is being measured)."""
    out = []
    seen = set()
    for m in MEASUREMENT_CONTEXT_RE.finditer(text):
        context = m.group(1).strip(" ,;:-")
        value = m.group(2)
        unit = m.group(3).lower()
        if unit == "percent":
            unit = "%"
        key = (context.lower(), value, unit)
        if key in seen:
            continue
        seen.add(key)
        out.append({"context": context, "value": value, "unit": unit})
        if len(out) >= 12:
            break
    return out


def extract_glossary(text: str) -> list[dict]:
    """Find glossary terms present in the report; return plain definitions."""
    lower = text.lower()
    found = []
    seen = set()
    # Longest terms first so "pleural effusion" beats "effusion".
    for term in sorted(GLOSSARY, key=len, reverse=True):
        if term in seen:
            continue
        # Word-boundary match.
        if re.search(rf"\b{re.escape(term)}\b", lower):
            # Skip if a longer already-found term contains this one.
            if any(term in f["term"] and term != f["term"] for f in found):
                continue
            found.append({"term": term, "definition": GLOSSARY[term]})
            seen.add(term)
        if len(found) >= 14:
            break
    return found


def _plain_rewrite(sent: str) -> str:
    """Light patient-friendly rewrite of a sentence (still extractive)."""
    s = sent.strip()
    s = re.sub(r"\s+", " ", s)
    # Capitalize first letter.
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


def summarize(text: str) -> dict:
    """Full deterministic analysis of a report."""
    text = text.strip()
    sections = split_sections(text)
    triage = compute_triage(text)

    # --- Key findings: rank sentences from findings/impression first ---
    priority_text = "\n".join(
        sections.get(k, "") for k in ("impression", "findings")
    ) or text
    sentences = split_sentences(priority_text)
    ranked = sorted(
        range(len(sentences)),
        key=lambda i: -_rank_sentence(sentences[i], i, len(sentences)),
    )
    key_findings = []
    for i in ranked[:6]:
        s = sentences[i].strip()
        if _BOILERPLATE_RE.match(s):
            continue
        if len(s) < 15:
            continue
        key_findings.append(_plain_rewrite(s))
        if len(key_findings) >= 4:
            break

    # --- Plain-language summary built from triage + key findings ---
    bullets = []
    # Lead with the triage headline.
    if triage["findings"]:
        top = triage["findings"][0]
        prefix = "Possible " if top["uncertain"] else ""
        bullets.append(f"{prefix}{top['reason']}.")
        for f in triage["findings"][1:4]:
            p = "Possible " if f["uncertain"] else ""
            line = f"{p}{f['reason']}."
            # Skip lines that duplicate an existing bullet modulo the prefix.
            if line in bullets or line.replace("Possible ", "") in [b.replace("Possible ", "") for b in bullets]:
                continue
            bullets.append(line)
    else:
        bullets.append("No red-flag findings were detected by the automated screen.")

    # Add a couple of key finding sentences for grounding.
    for kf in key_findings[:2]:
        if kf not in bullets and len(bullets) < 6:
            bullets.append(kf)

    measurements = extract_measurements(text)
    glossary = extract_glossary(text)

    return {
        "plain_summary": bullets,
        "key_findings": key_findings,
        "measurements": measurements,
        "glossary": glossary,
        "sections": {k: v for k, v in sections.items()},
        "triage": triage,
    }
