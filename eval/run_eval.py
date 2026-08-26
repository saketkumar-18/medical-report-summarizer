"""Safety evaluation harness for the MedSumm triage engine.

Metrics:
  1. Tier accuracy        — exact match and within-one-tier match
  2. Must-flag recall     — did we catch every finding the case requires?
                            (sensitivity; the critical safety metric)
  3. Must-NOT-flag rate   — did we wrongly flag a negated/absent finding?
                            (negation safety; false-alarm metric)
  4. Faithfulness         — every deterministic summary bullet must be grounded
                            (extractive engine should score 1.0)

Run:  python -m eval.run_eval
Writes: data/eval_results.json (served by GET /api/eval)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.summarizer import summarize  # noqa: E402
from app.triage import compute_triage  # noqa: E402
from app.lexicon import TIER_ORDER  # noqa: E402

EVAL_CASES = ROOT / "eval" / "cases.json"
OUT_PATH = ROOT / "data" / "eval_results.json"


def _flagged_reasons(triage: dict) -> str:
    return " | ".join(f["reason"].lower() for f in triage["findings"])


def _matched(must: str, triage: dict) -> bool:
    """A must_flag/must_not_flag phrase matches if any finding reason or the
    matched text mentions it (case-insensitive substring on reasons+text).
    Fuzzy fallback allows up to 4 chars between words so 'Lung-RADS 4' matches
    a reason written as 'Lung-RADS 3/4'."""
    hay = _flagged_reasons(triage) + " | " + " | ".join(
        f["matched_text"].lower() for f in triage["findings"]
    )
    m = must.lower()
    if m in hay:
        return True
    words = m.split()
    if len(words) > 1:
        pat = r".{0,4}".join(re.escape(w) for w in words)
        if re.search(pat, hay):
            return True
    return False


def run() -> dict:
    data = json.loads(EVAL_CASES.read_text(encoding="utf-8"))
    cases = data["cases"]

    results = []
    n_exact = 0
    n_within1 = 0
    must_flag_total = 0
    must_flag_hit = 0
    must_not_total = 0
    must_not_violation = 0
    faithfulness_sum = 0.0

    for case in cases:
        triage = compute_triage(case["report"])
        summary = summarize(case["report"])

        expected = case["tier"]
        got = triage["tier"]
        exact = got == expected
        within1 = abs(TIER_ORDER[got] - TIER_ORDER[expected]) <= 1

        flag_hits, flag_misses = [], []
        for mf in case.get("must_flag", []):
            must_flag_total += 1
            if _matched(mf, triage):
                must_flag_hit += 1
                flag_hits.append(mf)
            else:
                flag_misses.append(mf)

        notflag_violations = []
        for mnf in case.get("must_not_flag", []):
            must_not_total += 1
            if _matched(mnf, triage):
                must_not_violation += 1
                notflag_violations.append(mnf)

        # Faithfulness of deterministic bullets: every bullet is extracted from
        # the report or built from a detected finding reason, so we verify each
        # bullet's medical tokens appear in the source (approx via entity check).
        from app.llm import faithfulness_score
        bullets_text = " ".join(summary["plain_summary"])
        faith = faithfulness_score(case["report"], bullets_text)
        faithfulness_sum += faith

        if exact:
            n_exact += 1
        if within1:
            n_within1 += 1

        results.append(
            {
                "id": case["id"],
                "expected_tier": expected,
                "predicted_tier": got,
                "exact_match": exact,
                "within_one_tier": within1,
                "must_flag_hit": flag_hits,
                "must_flag_missed": flag_misses,
                "must_not_flag_violations": notflag_violations,
                "faithfulness": round(faith, 3),
                "n_findings": len(triage["findings"]),
            }
        )

    n = len(cases)
    metrics = {
        "n_cases": n,
        "tier_accuracy_exact": round(n_exact / n, 4),
        "tier_accuracy_within_one": round(n_within1 / n, 4),
        "must_flag_recall": round(must_flag_hit / must_flag_total, 4) if must_flag_total else 1.0,
        "must_flag_hit": must_flag_hit,
        "must_flag_total": must_flag_total,
        "negation_safety": round(1 - must_not_violation / must_not_total, 4) if must_not_total else 1.0,
        "must_not_flag_violations": must_not_violation,
        "must_not_flag_total": must_not_total,
        "mean_faithfulness": round(faithfulness_sum / n, 4),
    }

    out = {
        "description": "MedSumm safety evaluation results (deterministic engine).",
        "metrics": metrics,
        "cases": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    out = run()
    m = out["metrics"]
    print("=== MedSumm Safety Evaluation ===")
    print(f"Cases:                 {m['n_cases']}")
    print(f"Tier accuracy (exact): {m['tier_accuracy_exact']:.1%}")
    print(f"Tier accuracy (±1):    {m['tier_accuracy_within_one']:.1%}")
    print(f"Must-flag recall:      {m['must_flag_recall']:.1%}  ({m['must_flag_hit']}/{m['must_flag_total']})")
    print(f"Negation safety:       {m['negation_safety']:.1%}  ({m['must_not_flag_violations']} violations / {m['must_not_flag_total']})")
    print(f"Mean faithfulness:     {m['mean_faithfulness']:.3f}")
    print()
    for c in out["cases"]:
        if not c["exact_match"] or c["must_flag_missed"] or c["must_not_flag_violations"]:
            print(f"  [ISSUE] {c['id']}: expected={c['expected_tier']} got={c['predicted_tier']} "
                  f"missed={c['must_flag_missed']} violations={c['must_not_flag_violations']}")
