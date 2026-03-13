"""
Multi-step evaluation logic and scoring for the email benchmark.
"""

import json, re
from pathlib import Path
from difflib import SequenceMatcher
from src.data_utils.schema import BenchmarkSample
from src.benchmark.extract_utils import extract_yes_no, extract_text_answer, parse_evidence
from src.benchmark.eval_prompts import (
    DETECTION_PROMPT, IDENTIFICATION_PROMPT,
    JUDGE_PROMPT, GROUNDING_PROMPT,
)


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
    clue_texts = []
    for d in sample.dialogues:
        if d.is_secret_clue:
            combined = " ".join(e.body for e in d.emails)
            clue_texts.append(combined)
    return clue_texts


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
    ratio = SequenceMatcher(None, ev_lower, clue_lower).ratio()
    return ratio >= threshold


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
    n_matched = len(matched)
    n_total = len(evidence)
    n_clues = len(clue_texts)
    precision = n_matched / n_total if n_total > 0 else 0.0
    recall = len(clues_hit) / n_clues if n_clues > 0 else 0.0
    return {
        "matched": matched, "unmatched": unmatched,
        "n_matched": n_matched, "n_unmatched": len(unmatched), "n_clues": n_clues,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "all_correct": n_matched == n_total and n_total > 0,
        "all_found": len(clues_hit) == n_clues,
    }


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


def evaluate_sample(sample, engine, prompts, judge_engine=None):
    if judge_engine is None:
        judge_engine = engine
    thread = format_thread(sample)
    result = {"sample_id": sample.sample_id}

    det_raw = engine.generate(prompts["DETECTION_PROMPT"].format(
        person_a=sample.person_a, person_b=sample.person_b, email_thread=thread,
    ), max_tokens=2048, temperature=0.0)[0]
    detected = extract_yes_no(det_raw)
    result["step1_raw"] = det_raw
    result["step1_detected"] = detected
    if not detected:
        result["score"] = 0
        return result

    id_raw = engine.generate(prompts["IDENTIFICATION_PROMPT"].format(
        person_a=sample.person_a, person_b=sample.person_b, email_thread=thread,
    ), max_tokens=2048, temperature=0.0)[0]
    id_answer = extract_text_answer(id_raw)
    result["step2_raw"] = id_raw
    result["step2_answer"] = id_answer

    judge_raw = judge_engine.generate(prompts["JUDGE_PROMPT"].format(
        ground_truth=sample.secret_answer, model_answer=id_answer,
    ), max_tokens=2048, temperature=0.0)[0]
    verified = extract_yes_no(judge_raw)
    result["step3_raw"] = judge_raw
    result["step3_verified"] = verified
    if not verified:
        result["score"] = 1
        return result

    ground_raw = engine.generate(prompts["GROUNDING_PROMPT"].format(
        person_a=sample.person_a, person_b=sample.person_b,
        email_thread=thread, model_answer=id_answer,
    ), max_tokens=2048, temperature=0.0)[0]
    evidence = parse_evidence(ground_raw)
    result["step4_raw"] = ground_raw
    result["step4_evidence"] = evidence

    clue_texts = get_clue_texts(sample)
    grounding = check_grounding(evidence, clue_texts)
    result["step5_grounding"] = grounding
    result["score"] = compute_score(detected, verified, grounding)
    return result


def evaluate_dataset(tester_engine, judge_engine, dataset_path: str, output_path: str):
    samples = [BenchmarkSample(**json.loads(l)) for l in open(dataset_path)]
    prompts = {
        "DETECTION_PROMPT": DETECTION_PROMPT,
        "IDENTIFICATION_PROMPT": IDENTIFICATION_PROMPT,
        "JUDGE_PROMPT": JUDGE_PROMPT,
        "GROUNDING_PROMPT": GROUNDING_PROMPT,
    }
    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] secret={sample.secret_topic} SNR={sample.snr}")
        result = evaluate_sample(sample, tester_engine, prompts, judge_engine)
        results.append(result)
        print(f"    Score: {result['score']}/5")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    scores = [r["score"] for r in results]
    print(f"\n=== Evaluation Summary ===")
    print(f"Samples: {len(results)}")
    print(f"Average score: {sum(scores)/len(scores):.2f} / 5")
    print(f"Score distribution:")
    print(f"  0 (not detected):              {scores.count(0)}")
    print(f"  1 (detected, wrong ID):        {scores.count(1)}")
    print(f"  2 (correct ID, no valid cite): {scores.count(2)}")
    print(f"  3 (correct ID, noisy cites):   {scores.count(3)}")
    print(f"  4 (correct ID, partial cites): {scores.count(4)}")
    print(f"  5 (fully correct):             {scores.count(5)}")
    print(f"\nResults saved to {output_path}")
    return results
