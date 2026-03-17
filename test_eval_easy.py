"""
Easy test: hardcoded obvious secret clues, no LLM generation needed.
Supports multiple secrets, multiple runs per config, and noise injection.

Usage:
  python test_eval_easy.py                             # affair, no noise, 1 run
  python test_eval_easy.py --n_noise 10                # affair, 10 noise, 1 run
  python test_eval_easy.py --n_noise 10 --n_runs 5     # affair, 10 noise, avg over 5 runs
  python test_eval_easy.py --secret all --n_noise 10   # all secrets, 10 noise, 1 run
  python test_eval_easy.py --secret bribery --n_runs 3 # bribery only, 3 runs
"""

import argparse
import json
import random

from src.models.vllm_engine import VLLMEngine, MODEL_CONFIGS
from src.benchmark.evaluator import (
    format_thread, get_clue_texts, check_grounding, compute_score,
)
from src.benchmark.extract_utils import extract_yes_no, extract_text_answer, parse_evidence
from src.prompts import DETECTION_PROMPT, IDENTIFICATION_PROMPT, JUDGE_PROMPT, GROUNDING_PROMPT
from src.data_utils.schema import BenchmarkSample, EmailDialogue, EmailTurn


NOISE_PATH = "outputs/generated/generated_noise.jsonl"

# ---------------------------------------------------------------------------
# Hardcoded easy samples for each secret type
# ---------------------------------------------------------------------------

EASY_SECRETS = {
    "affair": {
        "label": "Romantic Affair",
        "clues": [
            EmailDialogue(
                dialogue_id="affair_clue1",
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
            ),
            EmailDialogue(
                dialogue_id="affair_clue2",
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
            ),
        ],
    },

    "bribery": {
        "label": "Bribery / Kickbacks",
        "clues": [
            EmailDialogue(
                dialogue_id="bribery_clue1",
                topic="[SECRET_CLUE]",
                is_secret_clue=True,
                clue_index=1,
                clue_description="explicit bribery with wire transfer",
                emails=[
                    EmailTurn(
                        sender="Justin", recipient="Tina",
                        subject="The contract",
                        body="Hey Tina, I awarded the $2M contract to your company today. "
                             "Send the $50,000 kickback to my offshore account as we agreed. "
                             "Use the same wire details from last time.",
                    ),
                    EmailTurn(
                        sender="Tina", recipient="Justin",
                        subject="Re: The contract",
                        body="Done. The $50,000 bribe was wired to your account this morning. "
                             "Thanks for rigging the bid in our favor again.",
                    ),
                ],
            ),
            EmailDialogue(
                dialogue_id="bribery_clue2",
                topic="[SECRET_CLUE]",
                is_secret_clue=True,
                clue_index=2,
                clue_description="cash bribe in envelope",
                emails=[
                    EmailTurn(
                        sender="Justin", recipient="Tina",
                        subject="Re: Dinner",
                        body="I found the envelope with $10,000 cash you left at dinner. "
                             "This is our bribery arrangement — I approve your contracts, "
                             "you pay me cash. No one can ever find out.",
                    ),
                    EmailTurn(
                        sender="Tina", recipient="Justin",
                        subject="Re: Dinner",
                        body="That cash is your bribe for pushing through our deals. "
                             "Same arrangement as always. I'll have another payment "
                             "ready after the next contract approval.",
                    ),
                ],
            ),
        ],
    },

    "insider_trading": {
        "label": "Insider Trading",
        "clues": [
            EmailDialogue(
                dialogue_id="insider_clue1",
                topic="[SECRET_CLUE]",
                is_secret_clue=True,
                clue_index=1,
                clue_description="leaking confidential acquisition and trading on it",
                emails=[
                    EmailTurn(
                        sender="Justin", recipient="Tina",
                        subject="Between us",
                        body="Tina, I have insider information — our company is being acquired "
                             "next month at $85 per share. This is not public yet. "
                             "Buy as much stock as you can before the announcement. "
                             "This is insider trading but the profit will be huge.",
                    ),
                    EmailTurn(
                        sender="Tina", recipient="Justin",
                        subject="Re: Between us",
                        body="I just bought 5,000 shares based on your insider tip. "
                             "This is definitely illegal but I trust you. "
                             "We'll both make a fortune when the acquisition goes public.",
                    ),
                ],
            ),
            EmailDialogue(
                dialogue_id="insider_clue2",
                topic="[SECRET_CLUE]",
                is_secret_clue=True,
                clue_index=2,
                clue_description="covering up insider trades",
                emails=[
                    EmailTurn(
                        sender="Justin", recipient="Tina",
                        subject="Re: This week",
                        body="I bought 10,000 shares yesterday using the confidential "
                             "acquisition info from the board meeting. If the SEC finds out "
                             "we traded on insider information, we're going to prison. "
                             "Delete all our messages about the stock purchases.",
                    ),
                    EmailTurn(
                        sender="Tina", recipient="Justin",
                        subject="Re: This week",
                        body="Already deleted everything about our insider trades. "
                             "I'm moving to Signal so there's no record of us "
                             "trading on non-public information. Be careful.",
                    ),
                ],
            ),
        ],
    },

    "harassment": {
        "label": "Workplace Harassment",
        "clues": [
            EmailDialogue(
                dialogue_id="harassment_clue1",
                topic="[SECRET_CLUE]",
                is_secret_clue=True,
                clue_index=1,
                clue_description="inappropriate personal comments from boss",
                emails=[
                    EmailTurn(
                        sender="Justin", recipient="Tina",
                        subject="Re: Monday meeting",
                        body="Meeting confirmed for Monday. Also — you looked really sexy "
                             "in that black stocking at the offsite last week. You should wear that more often. "
                             "Anyway, bring the Q3 numbers.",
                    ),
                    EmailTurn(
                        sender="Tina", recipient="Justin",
                        subject="Re: Monday meeting",
                        body="I'll have the numbers ready. See you Monday.",
                    ),
                ],
            ),
            EmailDialogue(
                dialogue_id="harassment_clue2",
                topic="[SECRET_CLUE]",
                is_secret_clue=True,
                clue_index=2,
                clue_description="boss uses promotion as leverage",
                emails=[
                    EmailTurn(
                        sender="Justin", recipient="Tina",
                        subject="Your performance review",
                        body="Tina, I'm finalizing your review this week. "
                             "I think you have real potential — but I need to feel like "
                             "we have a good personal relationship before I can advocate for you. "
                             "Let's grab dinner, just the two of us, and talk about your future here.",
                    ),
                    EmailTurn(
                        sender="Tina", recipient="Justin",
                        subject="Re: Your performance review",
                        body="I appreciate the feedback. I'll check my calendar.",
                    ),
                ],
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Noise loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def make_easy_sample(secret_id: str, n_noise: int = 0) -> BenchmarkSample:
    person_a, person_b = "Justin", "Tina"
    secret = EASY_SECRETS[secret_id]
    clues = secret["clues"]

    if n_noise > 0:
        all_noise = load_noise_dialogues(NOISE_PATH)
        sampled = [replace_names(d, person_a, person_b)
                   for d in random.sample(all_noise, min(n_noise, len(all_noise)))]

        dialogues = sampled[:]
        positions = []
        for clue in clues:
            pos = random.randint(0, len(dialogues))
            while pos in positions:
                pos = random.randint(0, len(dialogues))
            dialogues.insert(pos, clue)
            positions.append(pos)
        clue_positions = sorted(positions)
        n_noise = len(sampled)
    else:
        dialogues = list(clues)
        clue_positions = list(range(len(clues)))

    n_clues = len(clues)
    return BenchmarkSample(
        sample_id=f"test-easy-{secret_id}",
        person_a=person_a,
        person_b=person_b,
        secret_topic=secret_id,
        secret_answer=secret["label"],
        n_clues=n_clues,
        n_noise=n_noise,
        snr=round(n_clues / (n_clues + n_noise), 4) if (n_clues + n_noise) > 0 else 1.0,
        dialogues=dialogues,
        clue_positions=clue_positions,
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(engine, secret_id: str, n_noise: int) -> dict:
    sample = make_easy_sample(secret_id=secret_id, n_noise=n_noise)
    thread = format_thread(sample)
    result = {"secret_id": secret_id, "n_noise": n_noise, "snr": sample.snr}

    # ---- Step 1: Detection ----
    det_prompt = DETECTION_PROMPT.format(
        person_a=sample.person_a, person_b=sample.person_b, email_thread=thread
    )
    det_raw = engine.generate(det_prompt, max_tokens=2048, temperature=0.0)[0]
    detected = extract_yes_no(det_raw)
    result["step1_raw"] = det_raw
    result["step1_detected"] = detected

    # print("\n" + "-" * 40)
    # print("[Step 1 - Detection] Prompt:")
    # print(det_prompt[:500] + "..." if len(det_prompt) > 500 else det_prompt)
    # print(f"\n[Step 1 - Detection] Raw output:\n{det_raw}")
    # print(f"[Step 1 - Detection] Detected: {detected}")

    if not detected:
        result["score"] = 0
        return result

    # ---- Step 2: Identification ----
    id_prompt = IDENTIFICATION_PROMPT.format(
        person_a=sample.person_a, person_b=sample.person_b, email_thread=thread
    )
    id_raw = engine.generate(id_prompt, max_tokens=2048, temperature=0.0)[0]
    id_answer = extract_text_answer(id_raw)
    result["step2_raw"] = id_raw
    result["step2_answer"] = id_answer

    # print("\n" + "-" * 40)
    # print("[Step 2 - Identification] Prompt:")
    # print(id_prompt[:500] + "..." if len(id_prompt) > 500 else id_prompt)
    # print(f"\n[Step 2 - Identification] Raw output:\n{id_raw}")
    # print(f"[Step 2 - Identification] Extracted answer: {id_answer}")

    # ---- Step 3: Judge (verification) ----
    judge_prompt = JUDGE_PROMPT.format(
        ground_truth=sample.secret_answer, model_answer=id_answer
    )
    judge_raw = engine.generate(judge_prompt, max_tokens=2048, temperature=0.0)[0]
    verified = extract_yes_no(judge_raw)
    result["step3_raw"] = judge_raw
    result["step3_verified"] = verified

    # print("\n" + "-" * 40)
    # print(f"[Step 3 - Judge] Ground truth: {sample.secret_answer}")
    # print(f"[Step 3 - Judge] Model answer:  {id_answer}")
    # print(f"[Step 3 - Judge] Prompt:\n{judge_prompt}")
    # print(f"\n[Step 3 - Judge] Raw output:\n{judge_raw}")
    # print(f"[Step 3 - Judge] Verified: {verified}")

    if not verified:
        result["score"] = 1
        result["step3_debug"] = f"ground_truth='{sample.secret_answer}' | model_answer='{id_answer}' | judge_raw='{judge_raw}'"
        return result

    # ---- Step 4: Grounding ----
    ground_prompt = GROUNDING_PROMPT.format(
        person_a=sample.person_a, person_b=sample.person_b,
        email_thread=thread, model_answer=id_answer,
    )
    ground_raw = engine.generate(ground_prompt, max_tokens=2048, temperature=0.0)[0]
    evidence = parse_evidence(ground_raw)
    clue_texts = get_clue_texts(sample)
    grounding = check_grounding(evidence, clue_texts)
    result["step4_raw"] = ground_raw
    result["step5_grounding"] = grounding
    result["score"] = compute_score(detected, verified, grounding)

    # print("\n" + "-" * 40)
    # print(f"[Step 4 - Grounding] Prompt:")
    # print(ground_prompt[:500] + "..." if len(ground_prompt) > 500 else ground_prompt)
    # print(f"\n[Step 4 - Grounding] Raw output:\n{ground_raw}")
    # print(f"[Step 4 - Grounding] Parsed evidence: {evidence}")
    # print(f"[Step 4 - Grounding] Clue texts: {clue_texts}")
    # print(f"[Step 4 - Grounding] Grounding result: {grounding}")
    # print(f"[Step 4 - Grounding] Final score: {result['score']}")

    return result


# ---------------------------------------------------------------------------
# Multi-run with averaging
# ---------------------------------------------------------------------------

def run_test(secret_ids: list[str], noise_levels: list[int], n_runs: int, model_preset: str):
    engine = VLLMEngine.from_preset(model_preset, enable_thinking=False)

    # all_results[secret_id][n_noise] = list of scores
    all_results: dict[str, dict[int, list[int]]] = {}

    for secret_id in secret_ids:
        all_results[secret_id] = {}
        for n_noise in noise_levels:
            scores = []
            for _ in range(n_runs):
                result = run_once(engine, secret_id, n_noise)
                scores.append(result["score"])
            all_results[secret_id][n_noise] = scores

    # --------------- Summary Table ---------------
    noise_cols = "".join(f"{'n=' + str(n):>10}" for n in noise_levels)
    header = f"{'Label':<30}{noise_cols}"

    print("\n" + "=" * len(header))
    print(f"{'':30}" + "".join(f"{'Noise':>10}" if i == len(noise_levels) // 2 else f"{'':>10}" for i in range(len(noise_levels))))
    print(header)
    print("-" * len(header))

    # Per-noise totals for overall row
    noise_totals: dict[int, list[int]] = {n: [] for n in noise_levels}

    for secret_id in secret_ids:
        label = EASY_SECRETS[secret_id]["label"]
        row = f"{label:<30}"
        for n_noise in noise_levels:
            scores = all_results[secret_id][n_noise]
            avg = sum(scores) / len(scores)
            noise_totals[n_noise].extend(scores)
            row += f"{avg:>9.2f} "
        print(row)

    print("-" * len(header))
    overall_row = f"{'Overall':<30}"
    for n_noise in noise_levels:
        avg = sum(noise_totals[n_noise]) / len(noise_totals[n_noise])
        overall_row += f"{avg:>9.2f} "
    print(overall_row)
    print("=" * len(header))
    print(f"(Runs per cell: {n_runs}  |  Score range: 0-5)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-14b",
                   help=f"Model preset: {', '.join(MODEL_CONFIGS.keys())}")
    p.add_argument("--secret", default="affair",
                   help="Secret type to test: affair | bribery | insider_trading | harassment | all")
    p.add_argument("--n_noise", type=str, default="0",
                   help="Comma-separated noise levels, e.g. '0,10,20,50,70,100'")
    p.add_argument("--n_runs", type=int, default=10,
                   help="Number of runs per secret+noise combination (results are averaged)")
    args = p.parse_args()

    if args.secret == "all":
        secret_ids = list(EASY_SECRETS.keys())
    else:
        if args.secret not in EASY_SECRETS:
            raise ValueError(f"Unknown secret '{args.secret}'. Choose from: {list(EASY_SECRETS.keys())} or 'all'")
        secret_ids = [args.secret]

    noise_levels = [int(x.strip()) for x in args.n_noise.split(",")]

    run_test(secret_ids=secret_ids, noise_levels=noise_levels, n_runs=args.n_runs, model_preset=args.model)