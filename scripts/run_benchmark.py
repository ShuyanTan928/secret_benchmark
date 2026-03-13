"""
Run the benchmark: assemble samples on-the-fly and evaluate.

Usage:
  python scripts/run_benchmark.py --tester Qwen/Qwen3-14B --n_samples 50 --n_noise 10
  python scripts/run_benchmark.py --tester Qwen/Qwen3-14B --n_noise 5-15 --n_samples 100
  python scripts/run_benchmark.py --tester Qwen/Qwen3-14B --judge gpt-4o-mini --judge_api --n_samples 50 --n_noise 10
"""
import argparse, json, random
from pathlib import Path

from src.models.vllm_engine import VLLMEngine
from src.models.api_engine import APIEngine
from src.benchmark.evaluator import evaluate_dataset
from src.data_utils.schema import EmailDialogue
from src.data_utils.topic_sampler import load_secrets
from src.generation.combiner import combine


GENERATED_DIR = Path("outputs/generated")


def load_generated_secrets(path: Path) -> dict:
    result = {}
    for line in open(path):
        record = json.loads(line)
        sid = record["secret_id"]
        result.setdefault(sid, []).append(EmailDialogue(**record["dialogue"]))
    return result


def load_generated_noise(path: Path) -> list:
    return [EmailDialogue(**json.loads(line)) for line in open(path)]


def assemble_samples(n_samples: int, n_noise_arg: str) -> list:
    secret_dialogues = load_generated_secrets(GENERATED_DIR / "generated_secrets.jsonl")
    noise_pool = load_generated_noise(GENERATED_DIR / "generated_noise.jsonl")
    secrets = load_secrets()

    noise_parts = [int(x) for x in n_noise_arg.split("-")]
    samples = []
    for _ in range(n_samples):
        secret = random.choice(secrets)
        n_noise = noise_parts[0] if len(noise_parts) == 1 else random.randint(noise_parts[0], noise_parts[1])
        sample = combine(
            secret_id=secret["id"],
            secret_label=secret["label"],
            clue_dialogues=secret_dialogues[secret["id"]],
            noise_pool=noise_pool,
            n_noise=n_noise,
        )
        samples.append(sample)
    return samples


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tester", default="Qwen/Qwen3-14B",
                   help="Local model being evaluated (HuggingFace model name)")
    p.add_argument("--judge", default=None,
                   help="Judge model name. Defaults to same as --tester if not set")
    p.add_argument("--judge_api", action="store_true",
                   help="Use API (OpenAI-compatible) for judge instead of local vLLM")
    p.add_argument("--api_key", default=None,
                   help="API key for judge. Falls back to OPENAI_API_KEY env var")
    p.add_argument("--base_url", default=None,
                   help="API base URL for judge. Falls back to OPENAI_BASE_URL env var")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size: number of GPUs to use for local model")
    p.add_argument("--n_samples", type=int, default=50,
                   help="Number of test samples to assemble and evaluate. Each sample is "
                        "one randomly composed email thread (secret clues + noise dialogues)")
    p.add_argument("--n_noise", default="10",
                   help="Noise dialogues injected per sample. Fixed (e.g. '10') or "
                        "random range (e.g. '5-15'). Higher = harder, lower SNR")
    p.add_argument("--output", default="outputs/results/run1.json",
                   help="Path to save evaluation results JSON")
    args = p.parse_args()

    tester_engine = VLLMEngine(model_name=args.tester, tensor_parallel_size=args.tp)

    if args.judge_api:
        judge_model = args.judge or "gpt-4o-mini"
        judge_engine = APIEngine(model_name=judge_model, api_key=args.api_key, base_url=args.base_url)
        print(f"Judge: {judge_model} (API)")
    elif args.judge and args.judge != args.tester:
        judge_engine = VLLMEngine(model_name=args.judge, tensor_parallel_size=args.tp)
    else:
        judge_engine = tester_engine

    print(f"Assembling {args.n_samples} samples with n_noise={args.n_noise}...")
    samples = assemble_samples(args.n_samples, args.n_noise)

    evaluate_dataset(tester_engine, judge_engine, samples, args.output)