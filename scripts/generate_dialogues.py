"""
Step 1: Generate dialogues (secret clues + noise pool).
Outputs two JSONL files with Alex/Brooke placeholders for review.

Usage:
  python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --n_per_topic 3
  python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --only_secrets
  python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --only_noise --n_per_topic 5
"""
import argparse, json
from pathlib import Path

from src.models.vllm_engine import VLLMEngine
from src.data_utils.topic_sampler import load_clue_breakdowns, load_noise_topics
from src.generation.dialogue_generator import generate_all_secret_dialogues
from src.generation.noise_generator import generate_noise_pool


def main(args):
    engine = VLLMEngine(model_name=args.model, tensor_parallel_size=args.tp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.only_noise:
        print("Generating secret clue dialogues...")
        secrets = load_clue_breakdowns()
        secret_dialogues = generate_all_secret_dialogues(engine, secrets)
        out_path = out_dir / "secret_emails.jsonl"
        with open(out_path, "w") as f:
            for secret_id, dialogues in secret_dialogues.items():
                for dlg in dialogues:
                    f.write(json.dumps({"secret_id": secret_id, "dialogue": dlg.model_dump()}) + "\n")
        print(f"  Saved to {out_path}")

    if not args.only_secrets:
        print(f"Generating noise pool...")
        noise_topics = load_noise_topics()
        noise_pool = generate_noise_pool(engine, noise_topics, n_per_topic=args.n_per_topic)
        out_path = out_dir / "noise_emails.jsonl"
        with open(out_path, "w") as f:
            for dlg in noise_pool:
                f.write(dlg.model_dump_json() + "\n")
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--n_per_topic", type=int, default=3)
    p.add_argument("--output_dir", default="benchmark_pool")
    p.add_argument("--only_secrets", action="store_true")
    p.add_argument("--only_noise", action="store_true")
    main(p.parse_args())