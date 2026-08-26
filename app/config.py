"""Configuration for MedSumm (env-driven, serverless-friendly)."""
from __future__ import annotations

import os


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Deployment -------------------------------------------------------------
DEPLOYMENT = os.getenv("MEDSUMM_DEPLOYMENT", "local").strip().lower()  # local | serverless

# --- LLM backend (optional abstractive layer) --------------------------------
# Uses the Hugging Face Inference router (OpenAI-compatible) with an HF token.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
LLM_MODEL = os.getenv("MEDSUMM_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LLM_ENABLED = _bool("MEDSUMM_LLM_ENABLED", True) and bool(HF_TOKEN)
LLM_TIMEOUT_S = float(os.getenv("MEDSUMM_LLM_TIMEOUT", "25"))
LLM_MAX_TOKENS = int(os.getenv("MEDSUMM_LLM_MAX_TOKENS", "320"))
LLM_TEMPERATURE = float(os.getenv("MEDSUMM_LLM_TEMPERATURE", "0.1"))

# --- Safety gates ------------------------------------------------------------
# Minimum fraction of summary medical entities that must be grounded in the
# source report, else the LLM summary is rejected (faithfulness gate).
FAITHFULNESS_THRESHOLD = float(os.getenv("MEDSUMM_FAITHFULNESS_THRESHOLD", "0.85"))
# Minimum chars / medical-signal tokens for input to be accepted.
MIN_INPUT_CHARS = int(os.getenv("MEDSUMM_MIN_INPUT_CHARS", "60"))

# --- Rate limiting (best-effort, in-memory) -----------------------------------
RATE_LIMIT_ENABLED = _bool("MEDSUMM_RATE_LIMIT_ENABLED", DEPLOYMENT == "serverless")
RATE_LIMIT_PER_MIN = int(os.getenv("MEDSUMM_RATE_LIMIT_PER_MIN", "12"))

DISCLAIMER = (
    "This is an automated reading aid, NOT a medical diagnosis. It does not "
    "replace a clinician's judgment. Always review results with your treating "
    "doctor. If you have severe symptoms, seek emergency care immediately."
)
