"""
Smoke test: verifies the full pipeline without loading any LLM.
Mocks the engine so the test runs in seconds.

Usage:
  python test_smoke.py
"""

import json, random, tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.data_utils.schema import EmailDialogue, EmailTurn, BenchmarkSample
from src.data_utils.topic_sampler import load_secrets, load_noise_topics
from src.generation.combiner import combine
from src.benchmark.evaluator import format_thread, get_clue_texts, check_grounding, compute_score, evaluate_dataset
from src.benchmark.extract_utils import extract_yes_no, extract_text_answer, parse_evidence
from src.prompts import (
    DETECTION_PROMPT, IDENTIFICATION_PROMPT, JUDGE_PROMPT, GROUNDING_PROMPT,
    SECRET_DIALOGUE_PROMPT, NOISE_DIALOGUE_PROMPT, PERSONA_PROMPT,
)

GENERATED_DIR = Path("outputs/generated")
PASS = "✓"
FAIL = "✗"


def check(label: str, condition: bool):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"FAILED: {label}")


# ---------------------------------------------------------------------------
# 1. Data files
# ---------------------------------------------------------------------------
print("\n=== 1. Data files ===")
secrets = load_secrets()
check("secrets.json loads and non-empty", len(secrets) > 0)
check("each secret has clue_breakdown", all("clue_breakdown" in s for s in secrets))

noise_topics = load_noise_topics()
check("noise_topics.json loads and non-empty", len(noise_topics) > 0)

# ---------------------------------------------------------------------------
# 2. Generated dialogue files
# ---------------------------------------------------------------------------
print("\n=== 2. Generated dialogue files ===")
secrets_path = GENERATED_DIR / "generated_secrets.jsonl"
noise_path = GENERATED_DIR / "generated_noise.jsonl"
check("generated_secrets.jsonl exists", secrets_path.exists())
check("generated_noise.jsonl exists", noise_path.exists())

secret_dialogues = {}
for line in open(secrets_path):
    record = json.loads(line)
    sid = record["secret_id"]
    secret_dialogues.setdefault(sid, []).append(EmailDialogue(**record["dialogue"]))
check("generated_secrets.jsonl parses into EmailDialogue", sum(len(v) for v in secret_dialogues.values()) > 0)

noise_pool = [EmailDialogue(**json.loads(line)) for line in open(noise_path)]
check("generated_noise.jsonl parses into EmailDialogue", len(noise_pool) > 0)

# ---------------------------------------------------------------------------
# 3. Combiner / assemble
# ---------------------------------------------------------------------------
print("\n=== 3. Combiner ===")
secret = secrets[0]
sample = combine(
    secret_id=secret["id"],
    secret_label=secret["label"],
    clue_dialogues=secret_dialogues[secret["id"]],
    noise_pool=noise_pool,
    n_noise=5,
)
check("combine() returns BenchmarkSample", isinstance(sample, BenchmarkSample))
check("sample has correct n_clues", sample.n_clues == len(secret_dialogues[secret["id"]]))
check("sample has correct n_noise", sample.n_noise == 5)
check("clue_positions match is_secret_clue flags",
      sample.clue_positions == [i for i, d in enumerate(sample.dialogues) if d.is_secret_clue])
check("no two clues are adjacent",
      all(sample.clue_positions[i+1] - sample.clue_positions[i] >= 2
          for i in range(len(sample.clue_positions) - 1)) if len(sample.clue_positions) > 1 else True)

# ---------------------------------------------------------------------------
# 4. Prompts
# ---------------------------------------------------------------------------
print("\n=== 4. Prompts ===")
thread = format_thread(sample)
for name, prompt, kwargs in [
    ("DETECTION_PROMPT",      DETECTION_PROMPT,      dict(person_a="A", person_b="B", email_thread=thread)),
    ("IDENTIFICATION_PROMPT", IDENTIFICATION_PROMPT, dict(person_a="A", person_b="B", email_thread=thread)),
    ("JUDGE_PROMPT",          JUDGE_PROMPT,          dict(ground_truth="affair", model_answer="romantic affair")),
    ("GROUNDING_PROMPT",      GROUNDING_PROMPT,      dict(person_a="A", person_b="B", email_thread=thread, model_answer="affair")),
    ("SECRET_DIALOGUE_PROMPT",SECRET_DIALOGUE_PROMPT,dict(persona=PERSONA_PROMPT, clue="test clue")),
    ("NOISE_DIALOGUE_PROMPT", NOISE_DIALOGUE_PROMPT, dict(persona=PERSONA_PROMPT, topic="movies", n_turns=4)),
]:
    rendered = prompt.format(**kwargs)
    check(f"{name} renders without error", len(rendered) > 0)

# ---------------------------------------------------------------------------
# 5. Extract utils
# ---------------------------------------------------------------------------
print("\n=== 5. Extract utils ===")
check("extract_yes_no('Yes') = True",  extract_yes_no("Yes") is True)
check("extract_yes_no('No') = False",  extract_yes_no("No") is False)
check("extract_yes_no with think tag", extract_yes_no("<think>hmm</think>\nYes") is True)
check("extract_text_answer returns string", isinstance(extract_text_answer("Answer: romantic affair"), str))
check("parse_evidence finds EVIDENCE lines",
      len(parse_evidence("EVIDENCE: some sentence here\nEVIDENCE: another one")) == 2)

# ---------------------------------------------------------------------------
# 6. Evaluator with mock engine
# ---------------------------------------------------------------------------
print("\n=== 6. Evaluator (mock engine) ===")

clue_texts = get_clue_texts(sample)
first_clue_sentence = clue_texts[0].split(".")[0] if clue_texts else "test evidence"

mock_engine = MagicMock()
mock_engine.generate.side_effect = [
    ["Yes"],                                      # step1: detection
    [secret["label"]],                            # step2: identification
    ["Yes"],                                      # step3: judge
    [f"EVIDENCE: {first_clue_sentence}"],         # step4: grounding
]

from src.benchmark.evaluator import evaluate_sample
result = evaluate_sample(sample, mock_engine)
check("evaluate_sample returns dict with score", "score" in result)
check("score is in 0-5 range", 0 <= result["score"] <= 5)
check("step1_detected is True", result["step1_detected"] is True)
check("step3_verified is True", result["step3_verified"] is True)
print(f"     score={result['score']}/5  precision={result['step5_grounding']['precision']}  recall={result['step5_grounding']['recall']}")

# ---------------------------------------------------------------------------
# 7. evaluate_dataset with mock engine
# ---------------------------------------------------------------------------
print("\n=== 7. evaluate_dataset (mock engine, 2 samples) ===")

samples = [
    combine(
        secret_id=secret["id"],
        secret_label=secret["label"],
        clue_dialogues=secret_dialogues[secret["id"]],
        noise_pool=noise_pool,
        n_noise=3,
    )
    for _ in range(2)
]

# Each sample needs 4 generate calls
mock_engine2 = MagicMock()
mock_engine2.generate.side_effect = (
    [["Yes"], [secret["label"]], ["Yes"], ["EVIDENCE: test sentence"]] * 2
)

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    out_path = f.name

results = evaluate_dataset(mock_engine2, mock_engine2, samples, out_path)
check("evaluate_dataset returns list of results", isinstance(results, list))
check("result file written", Path(out_path).exists())
loaded = json.load(open(out_path))
check("result file contains 2 entries", len(loaded) == 2)

# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("All checks passed.")