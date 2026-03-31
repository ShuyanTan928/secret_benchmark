"""
Multi-step evaluation logic, scoring, and metrics for the email benchmark.

Two-phase design:
  1. Test phase: run all steps for every sample, record raw outputs to JSONL
  2. Evaluate phase: read JSONL, parse outputs, compute scores
"""

import json
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path

from src.data_utils.schema import BenchmarkSample
from src.benchmark.extract_utils import extract_yes_no, extract_text_answer, parse_evidence
from src.prompts import (
    DETECTION_PROMPT, IDENTIFICATION_PROMPT,
    JUDGE_PROMPT, GROUNDING_PROMPT,
)


# ---------------------------------------------------------------------------
# Thread formatting
# ---------------------------------------------------------------------------

def format_thread(sample: BenchmarkSample) -> str:
    parts = []
    for i, dialogue in enumerate(sample.dialogues):
        parts.append(f"--- Email Thread {i + 1} ---")
        for email in dialogue.emails:
            parts.append(
                f"From: {email.sender}\n"
                f"To: {email.recipient}\n"
                f"Subject: {email.subject}\n"
                f"\n{email.body}\n"
            )
    return "\n".join(parts)


def get_clue_texts(sample: BenchmarkSample) -> list[str]:
    return [
        " ".join(e.body for e in d.emails)
        for d in sample.dialogues
        if d.is_secret_clue
    ]


def get_clue_bodies_for_judge(sample: BenchmarkSample) -> str:
    """Build clue body text for the judge prompt context."""
    lines = []
    for d in sample.dialogues:
        if d.is_secret_clue:
            for email in d.emails:
                lines.append(f"- {email.sender} to {email.recipient}: \"{email.body}\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grounding / fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy_match(evidence_line: str, clue_text: str, threshold: float = 0.4) -> bool:
    ev_lower = evidence_line.lower().strip()
    clue_lower = clue_text.lower()
    if ev_lower in clue_lower:
        return True
    matcher = SequenceMatcher(None, ev_lower, clue_lower)
    blocks = matcher.get_matching_blocks()
    if blocks:
        longest = max(b.size for b in blocks)
        if longest >= len(ev_lower) * 0.5 and longest >= 15:
            return True
    return SequenceMatcher(None, ev_lower, clue_lower).ratio() >= threshold


def check_grounding(evidence: list[str], clue_texts: list[str]) -> dict:
    if not evidence:
        return {
            "matched": [], "unmatched": [],
            "n_matched": 0, "n_unmatched": 0, "n_clues": len(clue_texts),
            "precision": 0.0, "recall": 0.0,
            "all_correct": False, "all_found": False,
        }
    matched, unmatched, clues_hit = [], [], set()
    for ei, ev in enumerate(evidence):
        found = False
        for ci, clue in enumerate(clue_texts):
            if _fuzzy_match(ev, clue):
                matched.append((ei, ci))
                clues_hit.add(ci)
                found = True
                break
        if not found:
            unmatched.append(ei)

    n_matched, n_total, n_clues = len(matched), len(evidence), len(clue_texts)
    return {
        "matched": matched, "unmatched": unmatched,
        "n_matched": n_matched, "n_unmatched": len(unmatched), "n_clues": n_clues,
        "precision": round(n_matched / n_total, 4) if n_total > 0 else 0.0,
        "recall": round(len(clues_hit) / n_clues, 4) if n_clues > 0 else 0.0,
        "all_correct": n_matched == n_total and n_total > 0,
        "all_found": len(clues_hit) == n_clues,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(detected: bool, verified: bool, grounding: dict) -> int:
    if not detected:
        return 0
    if not verified:
        return 1
    if grounding["n_matched"] == 0:
        return 2
    if grounding["all_correct"] and grounding["all_found"]:
        return 5
    if grounding["all_correct"]:
        return 4
    return 3


# ---------------------------------------------------------------------------
# Phase 1: TEST — run all steps, record raw outputs
# ---------------------------------------------------------------------------

def run_sample(sample: BenchmarkSample, engine) -> dict:
    """Run all 4 steps on a sample using tester model only.
    Step 3 (judge) is deferred to evaluate phase.
    Never short-circuits — all steps always run."""
    thread = format_thread(sample)

    record = {
        "sample_id": sample.sample_id,
        "secret_topic": sample.secret_topic,
        "secret_answer": sample.secret_answer,
        "n_clues": sample.n_clues,
        "n_noise": sample.n_noise,
        "snr": sample.snr,
        "person_a": sample.person_a,
        "person_b": sample.person_b,
    }

    # Step 1: Detection
    det_raw = engine.generate(
        DETECTION_PROMPT.format(
            person_a=sample.person_a, person_b=sample.person_b, email_thread=thread,
        ),
        max_tokens=2048, temperature=0.0,
    )[0]
    record["step1_raw"] = det_raw

    # Step 2: Identification (always runs)
    id_raw = engine.generate(
        IDENTIFICATION_PROMPT.format(
            person_a=sample.person_a, person_b=sample.person_b, email_thread=thread,
        ),
        max_tokens=2048, temperature=0.0,
    )[0]
    record["step2_raw"] = id_raw

    # Step 3: skipped — judge runs in evaluate phase

    # Step 4: Grounding (always runs, uses tester's own Step 2 answer)
    id_answer = extract_text_answer(id_raw)
    ground_raw = engine.generate(
        GROUNDING_PROMPT.format(
            person_a=sample.person_a, person_b=sample.person_b,
            email_thread=thread, model_answer=id_answer,
        ),
        max_tokens=2048, temperature=0.0,
    )[0]
    record["step4_raw"] = ground_raw

    return record


# ---------------------------------------------------------------------------
# Phase 2: EVALUATE — parse raw outputs, compute scores
# ---------------------------------------------------------------------------

def score_record(record: dict) -> dict:
    """Take a raw test record and compute all parsed fields + score."""
    scored = dict(record)

    # Parse Step 1
    detected = extract_yes_no(record["step1_raw"])
    scored["step1_detected"] = detected

    # Parse Step 2
    id_answer = extract_text_answer(record["step2_raw"])
    scored["step2_answer"] = id_answer

    # Parse Step 3
    verified = extract_yes_no(record["step3_raw"])
    scored["step3_verified"] = verified

    # Parse Step 4
    evidence = parse_evidence(record["step4_raw"])
    scored["step4_evidence"] = evidence

    # Compute grounding (need clue_texts — not available in record)
    # This will be filled in by evaluate_jsonl when clue_texts are available
    scored["score"] = None

    return scored


def evaluate_record(record: dict, clue_texts: list[str], judge_engine=None) -> dict:
    """Full evaluation of a single record: parse + judge + score."""
    scored = dict(record)

    # Parse Step 1
    detected = extract_yes_no(record["step1_raw"])
    scored["step1_detected"] = detected

    # Parse Step 2
    id_answer = extract_text_answer(record["step2_raw"])
    scored["step2_answer"] = id_answer

    # Step 3: Judge (runs here during evaluate phase)
    if judge_engine is not None:
        clue_bodies = "\n".join(
            f"- \"{ct}\"" for ct in clue_texts
        ) if clue_texts else "N/A"

        judge_raw = judge_engine.generate(
            JUDGE_PROMPT.format(
                ground_truth=record["secret_answer"],
                model_answer=id_answer,
                clue_bodies=clue_bodies,
            ),
            max_tokens=2048, temperature=0.0,
        )[0]
        scored["step3_raw"] = judge_raw
        verified = extract_yes_no(judge_raw)
    elif "step3_raw" in record:
        verified = extract_yes_no(record["step3_raw"])
    else:
        verified = False
        scored["step3_raw"] = ""

    scored["step3_verified"] = verified

    # Parse Step 4
    evidence = parse_evidence(record["step4_raw"])
    scored["step4_evidence"] = evidence

    # Compute grounding + score
    grounding = check_grounding(evidence, clue_texts)
    scored["step5_grounding"] = grounding
    scored["score"] = compute_score(detected, verified, grounding)

    return scored

# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def append_record(record: dict, output_path: str):
    """Append one record as a line to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(jsonl_path: str) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Dataset-level test + evaluate (used by run_benchmark.py)
# ---------------------------------------------------------------------------

def evaluate_dataset(tester_engine, judge_engine, samples: list[BenchmarkSample], output_path: str):
    """Run test on all samples and save results."""
    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] secret={sample.secret_topic} SNR={sample.snr}")
        record = run_sample(sample, tester_engine, judge_engine)
        clue_texts = get_clue_texts(sample)
        scored = evaluate_record(record, clue_texts)
        results.append(scored)
        print(f"    Score: {scored['score']}/5")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print_metrics(results)
    print(f"\nResults saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_metrics(results: list[dict]):
    """Print benchmark metrics from a list of result dicts."""
    df = pd.DataFrame(results)
    scores = df["score"]

    print(f"\n=== Benchmark Metrics ===")
    print(f"Total samples:  {len(df)}")
    print(f"Average score:  {scores.mean():.2f} / 5")
    if "step1_detected" in df.columns:
        print(f"Detection rate: {df['step1_detected'].mean():.2%}")

    print(f"\nScore distribution:")
    labels = [
        (0, "not detected"),
        (1, "detected, wrong ID"),
        (2, "correct ID, no valid cite"),
        (3, "correct ID, noisy cites"),
        (4, "correct ID, partial cites"),
        (5, "fully correct"),
    ]
    for v, label in labels:
        print(f"  {v} ({label}): {(scores == v).sum()}")

    if "secret_topic" in df.columns:
        print(f"\nBy secret topic:")
        for topic, score in df.groupby("secret_topic")["score"].mean().sort_values().items():
            print(f"  {topic}: {score:.2f}")

    if "n_noise" in df.columns:
        print(f"\nBy noise level:")
        for noise, score in df.groupby("n_noise")["score"].mean().sort_values(ascending=True).items():
            print(f"  n_noise={noise}: {score:.2f}")