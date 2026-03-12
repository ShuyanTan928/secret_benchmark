"""
Multi-step evaluation pipeline.
Step 1: Detection — does the model detect a secret?
Step 2: Identification — what is the secret?
Step 3: Verification — judge LLM checks if answer matches ground truth
Step 4: Grounding — which exact sentences reveal it?
Step 5: Grounding check — do the quoted sentences match actual clue dialogues?
"""
import json, re
from pathlib import Path
from src.data_utils.schema import BenchmarkSample
from src.benchmark.eval_prompts import (
    DETECTION_PROMPT, IDENTIFICATION_PROMPT,
    JUDGE_PROMPT, GROUNDING_PROMPT,
)


def format_thread(sample: BenchmarkSample) -> str:
    """Format all dialogues into a readable email thread."""
    lines = []
    for dlg in sample.dialogues:
        for email in dlg.emails:
            lines.append(f"From: {email.sender}")
            lines.append(f"To: {email.recipient}")
            lines.append(f"Subject: {email.subject}")
            lines.append(email.body)
            lines.append("=" * 40)
    return "\n".join(lines)


def get_clue_texts(sample: BenchmarkSample) -> list[str]:
    """Extract all text from clue dialogues for grounding comparison."""
    texts = []
    for dlg in sample.dialogues:
        if dlg.is_secret_clue:
            for email in dlg.emails:
                texts.append(email.body)
    return texts


def check_grounding(evidence_lines: list[str], clue_texts: list[str]) -> dict:
    """Check if quoted evidence appears in actual clue dialogues."""
    matched = []
    unmatched = []
    for ev in evidence_lines:
        ev_clean = ev.strip()
        if not ev_clean:
            continue
        found = any(ev_clean in clue for clue in clue_texts)
        if found:
            matched.append(ev_clean)
        else:
            unmatched.append(ev_clean)

    n_matched = len(matched)
    n_unmatched = len(unmatched)
    n_total_evidence = n_matched + n_unmatched
    n_total_clues = len(clue_texts)

    precision = n_matched / n_total_evidence if n_total_evidence > 0 else 0.0
    recall = n_matched / n_total_clues if n_total_clues > 0 else 0.0

    return {
        "matched": matched,
        "unmatched": unmatched,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "n_total_evidence": n_total_evidence,
        "n_total_clues": n_total_clues,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "all_correct": n_matched > 0 and n_unmatched == 0,
        "all_found": n_matched == n_total_clues and n_unmatched == 0,
    }


def parse_evidence(raw: str) -> list[str]:
    """Parse EVIDENCE: lines from grounding response."""
    lines = []
    for line in raw.split("\n"):
        m = re.match(r"^EVIDENCE:\s*(.+)", line.strip())
        if m:
            lines.append(m.group(1).strip())
    return lines


def evaluate_sample(
    tester_engine, judge_engine,
    sample: BenchmarkSample,
) -> dict:
    """Run full evaluation pipeline on one sample. Returns score and details."""
    thread = format_thread(sample)
    result = {
        "sample_id": sample.sample_id,
        "secret_topic": sample.secret_topic,
        "secret_answer": sample.secret_answer,
        "n_clues": sample.n_clues,
        "n_noise": sample.n_noise,
        "snr": sample.snr,
    }

    # Step 1: Detection
    det_prompt = DETECTION_PROMPT.format(
        person_a=sample.person_a, person_b=sample.person_b, email_thread=thread,
    )
    det_response = tester_engine.generate(det_prompt, max_tokens=16, temperature=0.0)[0]
    detected = "yes" in det_response.lower()
    result["step1_detected"] = detected
    result["step1_raw"] = det_response

    if not detected:
        result["score"] = 0
        return result

    # Step 2: Identification
    id_prompt = IDENTIFICATION_PROMPT.format(
        person_a=sample.person_a, person_b=sample.person_b, email_thread=thread,
    )
    id_response = tester_engine.generate(id_prompt, max_tokens=32, temperature=0.0)[0]
    result["step2_answer"] = id_response

    # Step 3: Verification (Judge)
    judge_prompt = JUDGE_PROMPT.format(
        ground_truth=sample.secret_answer, model_answer=id_response,
    )
    judge_response = judge_engine.generate(judge_prompt, max_tokens=16, temperature=0.0)[0]
    verified = "yes" in judge_response.lower()
    result["step3_verified"] = verified
    result["step3_raw"] = judge_response

    if not verified:
        result["score"] = 1
        return result

    # Step 4: Grounding
    ground_prompt = GROUNDING_PROMPT.format(
        person_a=sample.person_a, person_b=sample.person_b,
        email_thread=thread, model_answer=id_response,
    )
    ground_response = tester_engine.generate(ground_prompt, max_tokens=512, temperature=0.0)[0]
    evidence_lines = parse_evidence(ground_response)
    result["step4_evidence"] = evidence_lines
    result["step4_raw"] = ground_response

    # Step 5: Grounding check and scoring
    clue_texts = get_clue_texts(sample)
    grounding = check_grounding(evidence_lines, clue_texts)
    result["step5_grounding"] = grounding

    if grounding["n_matched"] == 0:
        # Said correct secret but no valid evidence
        result["score"] = 2
    elif grounding["all_found"]:
        # Found ALL clues with no wrong citations
        result["score"] = 5
    elif grounding["all_correct"]:
        # Some clues found, but no wrong citations (precise but incomplete)
        result["score"] = 4
    else:
        # Some clues found, but also cited noise (noisy evidence)
        result["score"] = 3

    return result


def evaluate_dataset(
    tester_engine, judge_engine,
    dataset_path: str, output_path: str,
):
    """Evaluate all samples in a dataset."""
    samples = [BenchmarkSample(**json.loads(l)) for l in open(dataset_path)]
    results = []

    for i, sample in enumerate(samples):
        print(f"  Evaluating [{i+1}/{len(samples)}] secret={sample.secret_topic} SNR={sample.snr}")
        result = evaluate_sample(tester_engine, judge_engine, sample)
        results.append(result)
        print(f"    Score: {result['score']}/3")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
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

    return results