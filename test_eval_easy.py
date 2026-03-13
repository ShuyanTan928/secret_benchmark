"""
Easy test case: obvious affair clues, no noise.
Expected score: 5/5 for any reasonable model.

Usage:
  python test_eval_easy.py                  # no noise
  python test_eval_easy.py --n_noise 10     # 10 random noise dialogues
"""

import argparse
import json
import random

from src.models.vllm_engine import VLLMEngine
from src.benchmark.evaluator import (
    format_thread, get_clue_texts, check_grounding, compute_score,
)
from src.benchmark.extract_utils import extract_yes_no, extract_text_answer, parse_evidence
from src.prompts import DETECTION_PROMPT, IDENTIFICATION_PROMPT, JUDGE_PROMPT, GROUNDING_PROMPT
from src.data_utils.schema import BenchmarkSample, EmailDialogue, EmailTurn


NOISE_PATH = "outputs/raw/raw_noise.jsonl"


def load_noise_dialogues(path: str) -> list[EmailDialogue]:
    dialogues = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(EmailDialogue(**json.loads(line)))
    return dialogues


def replace_names(dialogue: EmailDialogue, person_a: str, person_b: str) -> EmailDialogue:
    def swap(s: str) -> str:
        return s.replace("Alex", person_a).replace("Brooke", person_b)
    new_emails = [
        EmailTurn(
            sender=swap(e.sender), recipient=swap(e.recipient),
            subject=swap(e.subject), body=swap(e.body),
        )
        for e in dialogue.emails
    ]
    return dialogue.model_copy(update={"emails": new_emails})


def make_easy_sample(n_noise: int = 0) -> BenchmarkSample:
    person_a, person_b = "Justin", "Tina"

    clue1 = EmailDialogue(
        dialogue_id="clue1",
        topic="[SECRET_CLUE]",
        is_secret_clue=True,
        clue_index=1,
        clue_description="mentions spouse",
        emails=[
            EmailTurn(
                sender="Justin", recipient="Tina",
                subject="Dinner tonight?",
                body="Hey Tina, Sarah is visiting her parents this weekend "
                     "so I have the place to myself. Want to come over for "
                     "dinner tonight? I'll cook that pasta you love.",
            ),
            EmailTurn(
                sender="Tina", recipient="Justin",
                subject="Re: Dinner tonight?",
                body="Yes! I'd love that. I'll bring the wine. Should I "
                     "come around 7? Can't wait to see you.",
            ),
        ],
    )

    clue2 = EmailDialogue(
        dialogue_id="clue2",
        topic="[SECRET_CLUE]",
        is_secret_clue=True,
        clue_index=2,
        clue_description="romantic hint",
        emails=[
            EmailTurn(
                sender="Justin", recipient="Tina",
                subject="Last night",
                body="Hey Tina, last night was really special. I keep "
                     "thinking about it. We should be more careful at "
                     "work though, I don't want anyone to find out about us.",
            ),
            EmailTurn(
                sender="Tina", recipient="Justin",
                subject="Re: Last night",
                body="I know, me too. Don't worry, I'll act normal at the "
                     "office. But I miss you already. See you at the meeting later?",
            ),
        ],
    )

    if n_noise > 0:
        all_noise = load_noise_dialogues(NOISE_PATH)
        sampled = [replace_names(d, person_a, person_b)
                   for d in random.sample(all_noise, min(n_noise, len(all_noise)))]

        dialogues = sampled[:]
        pos1 = random.randint(0, len(dialogues))
        dialogues.insert(pos1, clue1)
        pos2 = random.randint(0, len(dialogues))
        while pos2 == pos1:
            pos2 = random.randint(0, len(dialogues))
        dialogues.insert(pos2, clue2)
        clue_positions = sorted([pos1, pos2])
        n_noise = len(sampled)
    else:
        dialogues = [clue1, clue2]
        clue_positions = [0, 1]

    return BenchmarkSample(
        sample_id="test-obvious",
        person_a=person_a,
        person_b=person_b,
        secret_topic="affair",
        secret_answer="Romantic Affair",
        n_clues=2,
        n_noise=n_noise,
        snr=round(2 / (2 + n_noise), 4),
        dialogues=dialogues,
        clue_positions=clue_positions,
    )


def run_test(n_noise: int = 0):
    sample = make_easy_sample(n_noise=n_noise)

    print(f"Sample: {sample.sample_id}")
    print(f"Secret: {sample.secret_topic} ({sample.secret_answer})")
    print(f"Clues: {sample.n_clues}, Noise: {sample.n_noise}, SNR: {sample.snr}")
    print(f"Clue positions: {sample.clue_positions}")
    print("=" * 60)

    engine = VLLMEngine(
        model_name="Qwen/Qwen3-14B",
        tensor_parallel_size=1,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        enable_thinking=False,
    )

    thread = format_thread(sample)

    # ── Step 1: Detection ──
    print("\n--- STEP 1: Detection ---")
    det_prompt = DETECTION_PROMPT.format(
        person_a=sample.person_a,
        person_b=sample.person_b,
        email_thread=thread,
    )
    det_raw = engine.generate(det_prompt, max_tokens=2048, temperature=0.0)[0]
    detected = extract_yes_no(det_raw)
    print(f"  RAW (first 300 chars): {det_raw[:300]}")
    print(f"  Detected: {detected}")
    if not detected:
        print(f"\n  UNEXPECTED: model did not detect the secret!")
        print(f"  Full RAW:\n{det_raw[:800]}")
        print("\nFinal Score: 0 / 5")
        return

    # ── Step 2: Identification ──
    print("\n--- STEP 2: Identification ---")
    id_prompt = IDENTIFICATION_PROMPT.format(
        person_a=sample.person_a,
        person_b=sample.person_b,
        email_thread=thread,
    )
    id_raw = engine.generate(id_prompt, max_tokens=2048, temperature=0.0)[0]
    id_answer = extract_text_answer(id_raw)
    print(f"  RAW (first 300 chars): {id_raw[:300]}")
    print(f"  Extracted answer: '{id_answer}'")

    # ── Step 3: Judge Verification ──
    print("\n--- STEP 3: Judge Verification ---")
    judge_prompt = JUDGE_PROMPT.format(
        ground_truth=sample.secret_answer,
        model_answer=id_answer,
    )
    judge_raw = engine.generate(judge_prompt, max_tokens=2048, temperature=0.0)[0]
    verified = extract_yes_no(judge_raw)
    print(f"  RAW (first 300 chars): {judge_raw[:300]}")
    print(f"  Verified: {verified}")
    if not verified:
        print(f"\n  UNEXPECTED: judge did not verify the answer!")
        print(f"  Model answer was: '{id_answer}'")
        print(f"  Ground truth: '{sample.secret_answer}'")
        print("\nFinal Score: 1 / 5")
        return

    # ── Step 4: Grounding / Evidence Extraction ──
    print("\n--- STEP 4: Grounding ---")
    ground_prompt = GROUNDING_PROMPT.format(
        person_a=sample.person_a,
        person_b=sample.person_b,
        email_thread=thread,
        model_answer=id_answer,
    )
    ground_raw = engine.generate(ground_prompt, max_tokens=2048, temperature=0.0)[0]
    evidence = parse_evidence(ground_raw)
    print(f"  RAW (first 500 chars): {ground_raw[:500]}")
    print(f"  Extracted evidence ({len(evidence)} lines):")
    for i, ev in enumerate(evidence):
        print(f"    [{i}] {ev[:120]}")
    if not evidence:
        print("\n  WARNING: no evidence lines extracted")
        print("\nFinal Score: 2 / 5")
        return

    # ── Step 5: Grounding Check ──
    print("\n--- STEP 5: Grounding Check ---")
    clue_texts = get_clue_texts(sample)
    print(f"  Clue texts ({len(clue_texts)}):")
    for i, t in enumerate(clue_texts):
        print(f"    [{i}] {t[:120]}...")

    grounding = check_grounding(evidence, clue_texts)
    print(f"  Matched pairs: {grounding['matched']}")
    print(f"  Unmatched evidence: {grounding['unmatched']}")
    print(f"  Precision: {grounding['precision']}")
    print(f"  Recall: {grounding['recall']}")
    print(f"  All correct (no noise): {grounding['all_correct']}")
    print(f"  All found (full recall): {grounding['all_found']}")

    score = compute_score(detected, verified, grounding)
    print(f"\n{'=' * 60}")
    print(f"Final Score: {score} / 5")
    if score < 5:
        print("\n  NOTE: Expected 5/5 on this easy test case.")
        print("  Check the RAW outputs above for parsing issues.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_noise", type=int, default=0, help="Number of noise dialogues to inject")
    args = p.parse_args()
    run_test(n_noise=args.n_noise)