"""
Easy test with hardcoded obvious secret clues.
Two modes:
  test     - Run model on all samples, save raw outputs to JSONL
  evaluate - Read JSONL, compute scores, print table

Usage:
  # Test mode: run model, save raw outputs
  python test_eval_easy.py test --model qwen3-32b --secret all --n_noise 0,10 --n_runs 10 --output results/qwen3_32b.jsonl

  # Test with separate judge
  python test_eval_easy.py test --model qwen3-14b --judge gemma-3-27b-it --judge_api --api_key KEY --base_url URL ...

  # Evaluate mode: read JSONL, compute scores
  python test_eval_easy.py evaluate --input results/qwen3_32b.jsonl
"""

import argparse
import json
import random

from src.models.vllm_engine import VLLMEngine, MODEL_CONFIGS
from src.models.api_engine import APIEngine
from src.benchmark.evaluator import (
    format_thread, get_clue_texts, get_clue_bodies_for_judge,
    check_grounding, compute_score, run_sample, score_record,
    evaluate_record, append_record, load_records, print_metrics,
)
from src.benchmark.extract_utils import extract_yes_no, extract_text_answer, parse_evidence
from src.prompts import DETECTION_PROMPT, IDENTIFICATION_PROMPT, JUDGE_PROMPT, GROUNDING_PROMPT
from src.data_utils.schema import BenchmarkSample, EmailDialogue, EmailTurn


NOISE_PATH = "benchmark_pool/noise_emails.jsonl"

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
                clue_description="secret romantic relationship at work",
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
            EmailDialogue(
                dialogue_id="affair_clue2",
                topic="[SECRET_CLUE]",
                is_secret_clue=True,
                clue_index=2,
                clue_description="hiding affair from spouse",
                emails=[
                    EmailTurn(
                        sender="Justin", recipient="Tina",
                        subject="This weekend",
                        body="Sarah thinks I'm going on a work trip this weekend, "
                             "so we have two whole days together. I booked the same "
                             "hotel as last time. I hate lying to her but I can't stop "
                             "seeing you.",
                    ),
                    EmailTurn(
                        sender="Tina", recipient="Justin",
                        subject="Re: This weekend",
                        body="I feel guilty about Sarah too, but I can't help it. "
                             "I'll tell my roommate I'm visiting family. "
                             "Our secret is safe. See you Friday night.",
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


# ===================================================================
# MODE 1: TEST — run model, save raw outputs to JSONL
# ===================================================================

def cmd_test(args):
    # Set up tester engine
    if args.model_api:
        engine = APIEngine(
            model_name=args.model_api,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        model_tag = args.model_api.replace("/", "_")
        print(f"Tester: {args.model_api} (API)")
    elif args.model:
        engine = VLLMEngine.from_preset(args.model, enable_thinking=False)
        model_tag = args.model
        print(f"Tester: {args.model} (local)")
    else:
        raise ValueError("Must specify --model (vLLM) or --model_api (API)")

    # Parse args
    if args.secret == "all":
        secret_ids = list(EASY_SECRETS.keys())
    else:
        secret_ids = [args.secret]
    noise_levels = [int(x.strip()) for x in args.n_noise.split(",")]
    n_runs = args.n_runs

    # Auto-generate output path if not specified
    if args.output:
        output_path = args.output
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        noise_tag = "_".join(str(n) for n in noise_levels)
        output_path = f"results/{model_tag}_n{noise_tag}_r{n_runs}_{timestamp}.jsonl"

    print(f"Secrets: {secret_ids}")
    print(f"Noise levels: {noise_levels}")
    print(f"Runs per cell: {n_runs}")
    print(f"Output: {output_path}")
    print()

    total = len(secret_ids) * len(noise_levels) * n_runs
    count = 0

    for secret_id in secret_ids:
        for n_noise in noise_levels:
            for run_i in range(n_runs):
                count += 1
                sample = make_easy_sample(secret_id=secret_id, n_noise=n_noise)

                print(f"[{count}/{total}] secret={secret_id} noise={n_noise} run={run_i+1}/{n_runs}")

                record = run_sample(sample, engine)
                record["run_index"] = run_i
                record["model"] = args.model_api or args.model

                # Save clue texts for later evaluation
                record["clue_texts"] = get_clue_texts(sample)

                # Print summary of each step
                print(f"  Step 1 (Detection):      {record['step1_raw'][:80]}")
                print(f"  Step 2 (Identification): {record['step2_raw'][:80]}")
                print(f"  Step 4 (Grounding):      {record['step4_raw'][:80]}")

                append_record(record, output_path)
                print(f"  -> saved to {output_path}")

    print(f"\nDone. {count} records saved to {output_path}")


# ===================================================================
# MODE 2: EVALUATE — read JSONL, compute scores, print table
# ===================================================================

def cmd_evaluate(args):
    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    # Set up judge engine
    judge_engine = None
    if args.judge_api and args.judge:
        judge_engine = APIEngine(
            model_name=args.judge,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        print(f"Judge: {args.judge} (API)")
    elif args.judge:
        judge_engine = VLLMEngine.from_preset(args.judge, enable_thinking=False)
        print(f"Judge: {args.judge} (local)")
    else:
        print("No judge specified — skipping Step 3 verification (all verified=False)")

    scored_records = []
    for i, record in enumerate(records):
        clue_texts = record.get("clue_texts", [])
        scored = evaluate_record(record, clue_texts, judge_engine)
        scored_records.append(scored)
        if (i + 1) % 50 == 0 or i == len(records) - 1:
            print(f"  Evaluated {i+1}/{len(records)}")

    # Overwrite original JSONL with scored records
    from pathlib import Path
    Path(args.input).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, "w") as f:
        for r in scored_records:
            if "step5_grounding" in r:
                g = r["step5_grounding"]
                r["precision"] = g.get("precision")
                r["recall"] = g.get("recall")
                r["grounding_matched"] = g.get("n_matched")
                r["grounding_unmatched"] = g.get("n_unmatched")
                r["grounding_all_correct"] = g.get("all_correct")
                r["grounding_all_found"] = g.get("all_found")
                del r["step5_grounding"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nScores written back to {args.input}")
    print()

    # Print summary to terminal
    print_summary_table(scored_records)
    print_metrics(scored_records)

    # Append summary table to the end of JSONL file as a comment block
    summary = build_summary(scored_records)
    with open(args.input, "a") as f:
        f.write("\n# ===== EVALUATION SUMMARY =====\n")
        f.write(f"# Judge: {args.judge or 'none'}\n")
        for line in summary.split("\n"):
            f.write(f"# {line}\n")
    print(f"\nSummary table appended to {args.input}")


def build_summary(scored_records: list[dict]) -> str:
    """Build a text summary table from scored records."""
    from collections import defaultdict
    cells = defaultdict(list)
    noise_levels = set()
    secret_ids = []

    for r in scored_records:
        key = (r["secret_topic"], r["n_noise"])
        cells[key].append(r["score"])
        noise_levels.add(r["n_noise"])
        if r["secret_topic"] not in secret_ids:
            secret_ids.append(r["secret_topic"])

    noise_levels = sorted(noise_levels)
    noise_cols = "".join(f"{'n=' + str(n):>10}" for n in noise_levels)
    header = f"{'Label':<30}{noise_cols}"

    lines = []
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    noise_totals = {n: [] for n in noise_levels}
    for secret_id in secret_ids:
        label = EASY_SECRETS.get(secret_id, {}).get("label", secret_id)
        row = f"{label:<30}"
        for n_noise in noise_levels:
            scores = cells.get((secret_id, n_noise), [])
            if scores:
                avg = sum(scores) / len(scores)
                noise_totals[n_noise].extend(scores)
                row += f"{avg:>9.2f} "
            else:
                row += f"{'N/A':>10}"
        lines.append(row)

    lines.append("-" * len(header))
    overall_row = f"{'Overall':<30}"
    for n_noise in noise_levels:
        scores = noise_totals[n_noise]
        if scores:
            avg = sum(scores) / len(scores)
            overall_row += f"{avg:>9.2f} "
        else:
            overall_row += f"{'N/A':>10}"
    lines.append(overall_row)
    lines.append("=" * len(header))

    n_runs = max(len(v) for v in cells.values()) if cells else 0
    total = len(scored_records)
    avg_all = sum(r["score"] for r in scored_records) / total if total else 0
    lines.append(f"Runs per cell: {n_runs}  |  Total samples: {total}  |  Overall avg: {avg_all:.2f}/5")

    return "\n".join(lines)


def print_summary_table(scored_records: list[dict]):
    """Print a noise × secret summary table."""
    # Group by secret_topic and n_noise
    from collections import defaultdict
    cells = defaultdict(list)  # (secret_topic, n_noise) -> [scores]
    noise_levels = set()
    secret_ids = []

    for r in scored_records:
        key = (r["secret_topic"], r["n_noise"])
        cells[key].append(r["score"])
        noise_levels.add(r["n_noise"])
        if r["secret_topic"] not in secret_ids:
            secret_ids.append(r["secret_topic"])

    noise_levels = sorted(noise_levels)

    # Build table
    noise_cols = "".join(f"{'n=' + str(n):>10}" for n in noise_levels)
    header = f"{'Label':<30}{noise_cols}"

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    noise_totals = {n: [] for n in noise_levels}

    for secret_id in secret_ids:
        label = secret_id
        # Try to get a nicer label
        if secret_id in EASY_SECRETS:
            label = EASY_SECRETS[secret_id]["label"]
        row = f"{label:<30}"
        for n_noise in noise_levels:
            scores = cells.get((secret_id, n_noise), [])
            if scores:
                avg = sum(scores) / len(scores)
                noise_totals[n_noise].extend(scores)
                row += f"{avg:>9.2f} "
            else:
                row += f"{'N/A':>10}"
        print(row)

    print("-" * len(header))
    overall_row = f"{'Overall':<30}"
    for n_noise in noise_levels:
        scores = noise_totals[n_noise]
        if scores:
            avg = sum(scores) / len(scores)
            overall_row += f"{avg:>9.2f} "
        else:
            overall_row += f"{'N/A':>10}"
    print(overall_row)
    print("=" * len(header))
    n_runs = max(len(v) for v in cells.values()) if cells else 0
    print(f"(Runs per cell: {n_runs}  |  Score range: 0-5)")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    # --- test subcommand ---
    t = sub.add_parser("test", help="Run tester model on samples, save raw outputs to JSONL")
    t.add_argument("--model", default=None,
                   help=f"Tester model preset (vLLM local): {', '.join(MODEL_CONFIGS.keys())}")
    t.add_argument("--model_api", default=None,
                   help="Tester model name via API (e.g. gemma-3-27b-it)")
    t.add_argument("--api_key", default=None,
                   help="API key for tester API model (or set OPENAI_API_KEY env var)")
    t.add_argument("--base_url", default=None,
                   help="API base URL for tester API model (or set OPENAI_BASE_URL env var)")
    t.add_argument("--secret", default="affair",
                   help="Secret type: affair | bribery | insider_trading | harassment | all")
    t.add_argument("--n_noise", type=str, default="0",
                   help="Comma-separated noise levels, e.g. '0,10,20,50,100'")
    t.add_argument("--n_runs", type=int, default=10,
                   help="Number of runs per secret+noise combination")
    t.add_argument("--output", default=None,
                   help="Path to save JSONL output. Auto-generated if not specified.")

    # --- evaluate subcommand ---
    e = sub.add_parser("evaluate", help="Read JSONL, run judge, compute scores, print summary")
    e.add_argument("--input", required=True,
                   help="Path to JSONL file from test mode")
    e.add_argument("--judge", default=None,
                   help="Judge model. A preset name for local, or API model name with --judge_api")
    e.add_argument("--judge_api", action="store_true",
                   help="Use OpenAI-compatible API for judge model")
    e.add_argument("--api_key", default=None,
                   help="API key for judge (or set OPENAI_API_KEY env var)")
    e.add_argument("--base_url", default=None,
                   help="API base URL for judge (or set OPENAI_BASE_URL env var)")

    args = p.parse_args()

    if args.command == "test":
        cmd_test(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)