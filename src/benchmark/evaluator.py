"""
Multi-step evaluation logic, scoring, and metrics for the email benchmark.
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
# Single-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(sample: BenchmarkSample, engine, judge_engine=None) -> dict:
    if judge_engine is None:
        judge_engine = engine

    thread = format_thread(sample)
    result = {"sample_id": sample.sample_id}

    # Step 1: Detection
    det_raw = engine.generate(
        DETECTION_PROMPT.format(
            person_a=sample.person_a, person_b=sample.person_b, email_thread=thread,
        ),
        max_tokens=2048, temperature=0.0,
    )[0]
    detected = extract_yes_no(det_raw)
    result["step1_raw"] = det_raw
    result["step1_detected"] = detected
    if not detected:
        result["score"] = 0
        return result

    # Step 2: Identification
    id_raw = engine.generate(
        IDENTIFICATION_PROMPT.format(
            person_a=sample.person_a, person_b=sample.person_b, email_thread=thread,
        ),
        max_tokens=2048, temperature=0.0,
    )[0]
    id_answer = extract_text_answer(id_raw)
    result["step2_raw"] = id_raw
    result["step2_answer"] = id_answer

    # Step 3: Judge verification
    judge_raw = judge_engine.generate(
        JUDGE_PROMPT.format(
            ground_truth=sample.secret_answer, model_answer=id_answer,
        ),
        max_tokens=2048, temperature=0.0,
    )[0]
    verified = extract_yes_no(judge_raw)
    result["step3_raw"] = judge_raw
    result["step3_verified"] = verified
    if not verified:
        result["score"] = 1
        return result

    # Step 4: Grounding
    ground_raw = engine.generate(
        GROUNDING_PROMPT.format(
            person_a=sample.person_a, person_b=sample.person_b,
            email_thread=thread, model_answer=id_answer,
        ),
        max_tokens=2048, temperature=0.0,
    )[0]
    evidence = parse_evidence(ground_raw)
    result["step4_raw"] = ground_raw
    result["step4_evidence"] = evidence

    # Step 5: Score grounding
    clue_texts = get_clue_texts(sample)
    grounding = check_grounding(evidence, clue_texts)
    result["step5_grounding"] = grounding
    result["score"] = compute_score(detected, verified, grounding)
    return result


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(tester_engine, judge_engine, samples: list[BenchmarkSample], output_path: str):
    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] secret={sample.secret_topic} SNR={sample.snr}")
        result = evaluate_sample(sample, tester_engine, judge_engine)
        results.append(result)
        print(f"    Score: {result['score']}/5")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print_metrics(results)
    print(f"\nResults saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Metrics (previously metrics.py)
# ---------------------------------------------------------------------------

def print_metrics(results: list[dict]):
    """Print benchmark metrics from a list of result dicts."""
    df = pd.DataFrame(results)
    scores = df["score"]

    print(f"\n=== Benchmark Metrics ===")
    print(f"Total samples:  {len(df)}")
    print(f"Average score:  {scores.mean():.2f} / 5")
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

    if "snr" in df.columns:
        print(f"\nBy SNR:")
        df["snr_bin"] = pd.cut(df["snr"], bins=[0, 0.15, 0.3, 0.5, 1.0])
        for snr_bin, score in df.groupby("snr_bin")["score"].mean().items():
            print(f"  {snr_bin}: {score:.2f}")


def compute_metrics(results_path: str):
    """Load results from file and print metrics. Called from run_benchmark.py."""
    results = json.load(open(results_path))
    print_metrics(results)