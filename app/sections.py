"""Report sectioning: split free-text radiology/pathology reports into sections.

Handles common header conventions (FINDINGS:, IMPRESSION:, "Clinical History",
etc.) and falls back to a single 'findings' section when no headers exist.
"""
from __future__ import annotations

import re

from .lexicon import SECTION_ALIASES

# Build one big alternation of every alias, matched at line start or after
# whitespace, optionally followed by a colon. Longer aliases first so
# "clinical history" wins over "clinical".
_ALL_ALIASES: list[tuple[str, str]] = []
for canon, aliases in SECTION_ALIASES.items():
    for a in aliases:
        _ALL_ALIASES.append((a, canon))
_ALL_ALIASES.sort(key=lambda t: -len(t[0]))

_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?P<header>"
    + "|".join(re.escape(a) for a, _ in _ALL_ALIASES)
    + r")\s*(?::\s*|\n|$)",
    re.IGNORECASE,
)

_CANON_BY_ALIAS = {a.lower(): c for a, c in _ALL_ALIASES}


def split_sections(text: str) -> dict[str, str]:
    """Return {canonical_section: body}. Unknown headers -> 'findings'."""
    text = text.strip()
    matches = list(_HEADER_RE.finditer(text))

    if not matches:
        return {"findings": text}

    sections: dict[str, list[str]] = {}
    # Text before the first header is preamble (often clinical history).
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.setdefault("clinical_history", []).append(preamble)

    for i, m in enumerate(matches):
        canon = _CANON_BY_ALIAS[m.group("header").lower()]
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.setdefault(canon, []).append(body)

    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def split_sentences(text: str) -> list[str]:
    """Naive-but-robust sentence splitter that respects common abbreviations."""
    # Protect abbreviations with periods.
    protected = text
    abbrevs = ["e.g", "i.e", "vs", "Dr", "Mr", "Mrs", "Ms", "No", "no", "approx", "cm", "mm", "mg", "ml", "cc", "kg"]
    placeholders = {}
    for i, ab in enumerate(abbrevs):
        token = f"\x00ABBR{i}\x00"
        placeholders[token] = ab
        protected = re.sub(rf"\b{re.escape(ab)}\.", token, protected)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
    out = []
    for p in parts:
        for token, ab in placeholders.items():
            p = p.replace(token, ab + ".")
        p = p.strip()
        if p:
            out.append(p)
    return out
