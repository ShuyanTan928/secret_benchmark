"""
Step 1: Generate raw dialogues (secret clues + noise pool).
Outputs two JSONL files with PERSON_A/PERSON_B placeholders for manual review.

Usage:
  python scripts/generate_raw.py --model Qwen/Qwen3-14B --n_per_topic 1
"""
import argparse, json
from pathlib import Path
from tqdm import tqdm

from src.models.vllm_engine import VLLMEngine
from src.data_utils.topic_sampler import load_secrets, load_noise_topics
from src.generation.dialogue_generator import generate_all_secret_dialogues
from src.generation.noise_generator import generate_noise_pool


def main(args):
    engine = VLLMEngine(model_name=args.model, tensor_parallel_size=args.tp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    secrets = load_secrets()
    noise_topics = load_noise_topics()

    # Step 1: Generate secret clue dialogues
    if not args.only_noise:
        print("Generating secret clue dialogues...")
        secret_dialogues = generate_all_secret_dialogues(engine, secrets)

        secret_path = out_dir / "raw_secrets.jsonl"
        with open(secret_path, "w") as f:
            for secret_id, dialogues in secret_dialogues.items():
                for dlg in dialogues:
                    record = {"secret_id": secret_id, "dialogue": dlg.model_dump()}
                    f.write(json.dumps(record) + "\n")
        print(f"  Secret dialogues saved to {secret_path}")

    # Step 2: Generate noise pool
    if not args.only_secrets:
        print(f"Generating noise pool ({len(noise_topics)} topics x {args.n_per_topic} each)...")
        noise_pool = generate_noise_pool(engine, noise_topics, n_per_topic=args.n_per_topic)

        noise_path = out_dir / "raw_noise.jsonl"
        with open(noise_path, "w") as f:
            for dlg in noise_pool:
                f.write(dlg.model_dump_json() + "\n")
        print(f"  Noise dialogues saved to {noise_path}")

    print(f"\nDone! Review the files in {out_dir}/ before running assemble_dataset.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--n_per_topic", type=int, default=3)
    p.add_argument("--output_dir", default="outputs/raw")
    p.add_argument("--only_secrets", action="store_true", help="Only generate secret clue dialogues, skip noise")
    p.add_argument("--only_noise", action="store_true", help="Only generate noise dialogues, skip secrets")
    main(p.parse_args())