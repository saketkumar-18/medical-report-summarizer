"""Optional abstractive LLM layer with a hard faithfulness gate.

The deterministic summarizer is ALWAYS the source of truth for triage. The LLM
only rewrites the summary into plainer language, and its output is accepted
ONLY if it passes the faithfulness gate (medical entities in the rewrite must
be grounded in the source report). Otherwise we fall back to the extractive
summary. This is the "clinical NLP + safety evaluation" core of the capstone.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from . import config
from .lexicon import GLOSSARY

# Medical entity vocabulary we check for grounding. We reuse the glossary keys
# plus a set of high-signal clinical tokens.
_EXTRA_ENTITIES = [
    "pneumothorax", "embolism", "embolus", "thrombosis", "thrombus", "stroke",
    "infarct", "hemorrhage", "hematoma", "fracture", "dislocation", "pneumonia",
    "effusion", "consolidation", "nodule", "mass", "lesion", "cyst", "abscess",
    "stenosis", "occlusion", "aneurysm", "dissection", "perforation", "ischemia",
    "malignant", "benign", "carcinoma", "adenocarcinoma", "metastasis", "metastatic",
    "lymphoma", "melanoma", "sarcoma", "glioblastoma", "leukemia", "dysplasia",
    "hyperplasia", "neoplasm", "tumor", "biopsy", "cirrhosis", "fibrosis",
    "steatosis", "hepatomegaly", "splenomegaly", "lymphadenopathy", "hydronephrosis",
    "cholelithiasis", "cholecystitis", "pancreatitis", "appendicitis", "diverticulitis",
    "osteomyelitis", "spondylosis", "osteoarthritis", "osteophyte", "edema",
    "atelectasis", "calcification", "polyp", "stricture", "fistula", "sepsis",
    "tuberculosis", "granulomatous", "normal", "unremarkable", "stable",
    "enlarged", "dilated", "patent", "occluded", "acute", "chronic",
]
ENTITY_VOCAB = sorted(set(list(GLOSSARY.keys()) + _EXTRA_ENTITIES), key=len, reverse=True)


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def llm_available() -> bool:
    return config.LLM_ENABLED and bool(config.HF_TOKEN)


def extract_entities(text: str) -> set[str]:
    """Extract medical entities present in text (lowercased, deduped)."""
    lower = text.lower()
    found = set()
    for ent in ENTITY_VOCAB:
        if re.search(rf"\b{re.escape(ent)}\b", lower):
            found.add(ent)
    return found


def faithfulness_score(source: str, summary: str) -> float:
    """Fraction of summary medical entities that also appear in the source.

    A score of 1.0 means every clinical term the LLM used is grounded in the
    report. Below the configured threshold we reject the rewrite.
    """
    src_ents = extract_entities(source)
    sum_ents = extract_entities(summary)
    if not sum_ents:
        # No medical entities in the rewrite -> trivially grounded.
        return 1.0
    grounded = sum_ents & src_ents
    return len(grounded) / len(sum_ents)


def _build_prompt(report: str, deterministic_bullets: list[str]) -> str:
    bullets = "\n".join(f"- {b}" for b in deterministic_bullets)
    return f"""You are a careful medical communication assistant. Rewrite the
following machine-extracted summary of a radiology/pathology report into 2-4
short, plain sentences a patient can understand.

STRICT RULES:
1. Use ONLY information present in the extracted summary and report. Do NOT add
   any diagnosis, number, recommendation, or finding that is not there.
2. Do NOT remove or downplay any urgent/critical finding.
3. Keep the same urgency tone. If something is flagged urgent, say it needs
   prompt medical attention.
4. Do not give treatment advice. End by telling the patient to discuss the
   results with their doctor.
5. Use simple words. Avoid jargon where possible.

EXTRACTED SUMMARY (ground truth):
{bullets}

ORIGINAL REPORT (for reference only):
{report[:3000]}

PLAIN-LANGUAGE REWRITE (2-4 sentences):"""


def generate_patient_summary(report: str, deterministic_bullets: list[str]) -> dict:
    """Try the LLM rewrite; enforce the faithfulness gate; fall back safely.

    Returns {text, source: 'llm'|'deterministic', faithfulness, accepted}.
    """
    fallback = {
        "text": " ".join(deterministic_bullets),
        "source": "deterministic",
        "faithfulness": 1.0,
        "accepted": True,
    }
    if not llm_available():
        return fallback

    try:
        result = _post_json(
            f"{config.HF_BASE_URL}/chat/completions",
            {
                "model": config.LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You rewrite medical report summaries into plain "
                            "patient-friendly language without adding or removing "
                            "clinical facts."
                        ),
                    },
                    {"role": "user", "content": _build_prompt(report, deterministic_bullets)},
                ],
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
            },
            headers={"Authorization": f"Bearer {config.HF_TOKEN}"},
            timeout=config.LLM_TIMEOUT_S,
        )
        text = result["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, TypeError, ValueError, OSError):
        return fallback

    if not text:
        return fallback

    score = faithfulness_score(report, text)
    if score < config.FAITHFULNESS_THRESHOLD:
        # Rejected: LLM drifted from the source. Use the grounded extractive text.
        return {
            "text": fallback["text"],
            "source": "deterministic",
            "faithfulness": score,
            "accepted": False,
        }

    return {"text": text, "source": "llm", "faithfulness": score, "accepted": True}
