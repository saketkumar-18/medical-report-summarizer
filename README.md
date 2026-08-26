# 🩺 MedSumm — Medical Report Summarizer + Triage Assistant

Summarize radiology and pathology reports in plain language, flag urgency, and
explain the jargon — built as a **clinical NLP + safety evaluation** capstone.

**Live:** https://medical-report-summarizer-phi.vercel.app

> ⚠️ **Not a medical device.** This is an automated reading aid for patients.
> It does not replace a clinician's judgment and must not be used to make
> medical decisions. Synthetic evaluation data only — no real patient information.

---

## What it does

Paste a radiology or pathology report and get:

1. **Automated triage level** — one of four urgency tiers with concrete advice:
   - 🚨 **Critical** — seek immediate medical attention (PE, stroke, hemorrhage, dissection, torsion, cauda equina…)
   - ⚠️ **Urgent** — contact your doctor within 24–48h (pneumonia, fractures, malignancy, BI-RADS 4/5…)
   - 📅 **Routine** — discuss at next appointment (nodules, cysts, degenerative changes…)
   - ✅ **Benign** — normal / no urgent findings
2. **Plain-language summary** — extractive, grounded in the source text.
3. **Optional AI rewrite** — a safety-gated LLM layer that rewrites the summary
   into plainer language, rejected if it drifts from the source.
4. **Jargon glossary** — ~150 clinical terms defined in plain words, matched to
   the terms actually present in your report.
5. **Measurements** — sizes/percentages extracted with context.

## Architecture

```
                 ┌────────────────────────────────────────────┐
 report text ──▶ │  Input guards (length, charset, injection) │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────────────────────────────────────┐
                 │  Section parser (history/findings/…)        │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────────────────────────────────────┐
                 │  Red-flag detector (~70 curated patterns)   │
                 │  + ConNeg-style negation/uncertainty engine │
                 │  negated → drop · uncertain → escalate      │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────────────────────────────────────┐
                 │  Triage engine (max severity + benign rule) │
                 │  Extractive summarizer + glossary + meas.   │
                 └──────────────────┬─────────────────────────┘
                                    ▼
                 ┌────────────────────────────────────────────┐
                 │  OPTIONAL LLM rewrite → faithfulness gate   │
                 │  (reject if medical terms not grounded)     │
                 └────────────────────────────────────────────┘
```

**Design principle:** the deterministic engine is always the source of truth for
triage. The LLM is cosmetic and gated — the app works fully with no LLM at all.

## Safety design

- **Negation-aware:** "no pneumothorax" is never flagged. A ConNeg-style
  assertion classifier labels every finding as *affirmed / negated / uncertain*.
- **Conservative escalation:** "cannot exclude pulmonary embolism" is treated as
  a possible emergency (uncertain → escalated, marked "possible"). Missing a
  real emergency is worse than over-flagging.
- **Explicit benign clearance dominates** routine mentions — e.g. "nodule" in
  the clinical history of a biopsy that came back benign does not escalate.
- **Faithfulness gate:** the LLM rewrite is accepted only if ≥85% of its
  medical entities are grounded in the source report; otherwise the grounded
  extractive summary is shown and the rejection is surfaced to the user.
- **Input guards:** prompt-injection patterns, control characters, and
  malformed input are rejected before processing. Rate-limited in serverless.

## Safety evaluation (capstone angle)

A 30-case labeled synthetic set (`eval/cases.json`) covers all four tiers plus
negation traps ("no PE", "no fracture") and uncertainty traps ("cannot exclude
PE"). `eval/run_eval.py` scores the engine on every change:

| Metric | Score |
|---|---|
| Triage tier accuracy (exact) | **100%** (30/30) |
| Within ±1 tier | **100%** |
| Must-flag recall (sensitivity) | **100%** (26/26 critical findings caught) |
| Negation safety (no false alarms) | **100%** (0/24 violations) |
| Mean summary faithfulness | **91.2%** |

Results are bundled and served live at `/api/eval` and rendered in the
**Safety evaluation** tab. Regenerate with `python -m eval.run_eval`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness + config (LLM enabled, thresholds) |
| `/api/analyze` | POST | `{report, use_llm}` → triage + summary + glossary |
| `/api/samples` | GET | 10 sample reports across all tiers |
| `/api/eval` | GET | Bundled safety-evaluation results |

## Run locally

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8321 --reload
# or: run_local.bat
```

Tests & eval:

```bash
python -m pytest tests/ -q      # 22 unit tests
python -m eval.run_eval         # 30-case safety evaluation
```

## Deploy (Vercel serverless)

```bash
vercel deploy --prod
```

`vercel.json` wires `api/index.py` (FastAPI ASGI) with `static/` + `data/`
bundled. Optional LLM layer: set `HF_TOKEN` in Vercel env vars.

## Project layout

```
app/
  config.py      env-driven config
  lexicon.py     red flags, urgency tiers, negation cues, glossary (~150 terms)
  negation.py    ConNeg-style assertion classifier
  sections.py    report section parser + sentence splitter
  triage.py      negation-aware red-flag detection + urgency scoring
  summarizer.py  extractive summarizer, measurements, glossary matching
  llm.py         safety-gated LLM rewrite + faithfulness gate
  main.py        FastAPI app, input guards, rate limiting
api/index.py     Vercel serverless entrypoint
eval/            30-case labeled safety set + evaluation harness
tests/           22 unit tests
static/          single-page frontend (analyzer, samples, eval dashboard)
data/            samples.json + generated eval_results.json
```

## Tech

Python 3.11 · FastAPI · Pydantic · zero-ML deterministic NLP core · optional
HF Inference LLM layer · vanilla JS frontend · Vercel serverless.
