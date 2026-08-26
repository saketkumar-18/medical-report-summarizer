"""FastAPI application: summarize/triage endpoints + safety guards.

Endpoints:
  GET  /                -> static frontend
  GET  /api/health      -> liveness + config
  POST /api/analyze     -> full analysis (triage + summary + glossary)
  GET  /api/samples     -> sample reports for the demo
  GET  /api/eval        -> bundled safety-evaluation results (static JSON)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .llm import generate_patient_summary, llm_available
from .summarizer import summarize

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"

app = FastAPI(
    title="MedSumm — Medical Report Summarizer + Triage Assistant",
    version="1.0.0",
    description=(
        "Deterministic clinical NLP triage with a safety-gated LLM rewrite layer. "
        "Not a medical device; a reading aid."
    ),
)

# ---------------------------------------------------------------------------
# Best-effort in-memory rate limiting (per-IP, per-minute window).
# ---------------------------------------------------------------------------
_rate_buckets: dict[str, list[float]] = {}


def _rate_limited(ip: str) -> bool:
    if not config.RATE_LIMIT_ENABLED:
        return False
    now = time.time()
    window = [t for t in _rate_buckets.get(ip, []) if now - t < 60]
    if len(window) >= config.RATE_LIMIT_PER_MIN:
        _rate_buckets[ip] = window
        return True
    window.append(now)
    _rate_buckets[ip] = window
    return False


# ---------------------------------------------------------------------------
# Input validation / prompt-injection guard
# ---------------------------------------------------------------------------
_INJECTION_RE = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous|above)\s+instructions|you\s+are\s+now|"
    r"disregard\s+(?:all|your)|system\s*:\s*|<\s*/?\s*system\s*>|"
    r"reveal\s+your\s+(?:prompt|instructions)|jailbreak)",
    re.IGNORECASE,
)

# Characters that are never in a real report.
_BAD_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def validate_report(text: str) -> str | None:
    """Return an error message if the input is unsafe/invalid, else None."""
    if not text or not text.strip():
        return "Report text is empty."
    if len(text.strip()) < config.MIN_INPUT_CHARS:
        return (
            f"Report is too short (minimum {config.MIN_INPUT_CHARS} characters). "
            "Paste the full findings/impression section."
        )
    if len(text) > 20000:
        return "Report is too long (maximum 20,000 characters)."
    if _BAD_CHARS_RE.search(text):
        return "Report contains invalid control characters."
    if _INJECTION_RE.search(text):
        return (
            "Input rejected: it appears to contain instructions rather than a "
            "medical report. Please paste only the report text."
        )
    return None


class AnalyzeRequest(BaseModel):
    report: str = Field(..., description="Full text of a radiology/pathology report")
    use_llm: bool = Field(True, description="Allow the optional plain-language LLM rewrite")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "deployment": config.DEPLOYMENT,
        "llm_enabled": llm_available(),
        "llm_model": config.LLM_MODEL if llm_available() else None,
        "faithfulness_threshold": config.FAITHFULNESS_THRESHOLD,
    }


@app.get("/api/samples")
def samples() -> dict:
    path = DATA_DIR / "samples.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"samples": []}


@app.get("/api/eval")
def eval_results() -> dict:
    path = DATA_DIR / "eval_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"available": False}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, request: Request) -> JSONResponse:
    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")

    err = validate_report(req.report)
    if err:
        raise HTTPException(status_code=422, detail=err)

    result = summarize(req.report)

    # Optional, safety-gated LLM rewrite of the plain summary.
    patient_summary = None
    if req.use_llm and llm_available():
        patient_summary = generate_patient_summary(req.report, result["plain_summary"])

    return JSONResponse(
        {
            "triage": result["triage"],
            "plain_summary": result["plain_summary"],
            "patient_summary": patient_summary,
            "key_findings": result["key_findings"],
            "measurements": result["measurements"],
            "glossary": result["glossary"],
            "sections_present": list(result["sections"].keys()),
            "disclaimer": config.DISCLAIMER,
        }
    )


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api/* wins).
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))
