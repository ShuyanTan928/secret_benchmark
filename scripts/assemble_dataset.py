"""
Step 2: Assemble benchmark dataset from raw dialogues.
Reads raw_secrets.jsonl and raw_noise.jsonl, combines them with random names.
No LLM needed — pure shuffling and name replacement.

Usage:
  python scripts/assemble_dataset.py --n_samples 50 --n_noise 8-15
"""
import argparse, json, random
from pathlib import Path
from tqdm import tqdm

from src.data_utils.schema import EmailDialogue
from src.data_utils.topic_sampler import load_secrets
from src.generation.combiner import combine


def load_raw_secrets(path: str) -> dict:
    """Load raw secret dialogues. Returns {secret_id: [EmailDialogue]}."""
    result = {}
    for line in open(path):
        record = json.loads(line)
        sid = record["secret_id"]
        dlg = EmailDialogue(**record["dialogue"])
        result.setdefault(sid, []).append(dlg)
    return result


def load_raw_noise(path: str) -> list:
    """Load raw noise dialogues."""
    return [EmailDialogue(**json.loads(line)) for line in open(path)]


def main(args):
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    secret_dialogues = load_raw_secrets(raw_dir / "raw_secrets.jsonl")
    noise_pool = load_raw_noise(raw_dir / "raw_noise.jsonl")
    secrets = load_secrets()

    print(f"Loaded {sum(len(v) for v in secret_dialogues.values())} secret dialogues")
    print(f"Loaded {len(noise_pool)} noise dialogues")

    with open(out_path, "w") as fout:
        for i in tqdm(range(args.n_samples), desc="Assembling samples"):
            secret = random.choice(secrets)
            noise_parts = [int(x) for x in args.n_noise.split("-")]
            if len(noise_parts) == 1:
                n_noise = noise_parts[0]
            else:
                n_noise = random.randint(noise_parts[0], noise_parts[1])

            sample = combine(
                secret_id=secret["id"],
                secret_label=secret["label"],
                clue_dialogues=secret_dialogues[secret["id"]],
                noise_pool=noise_pool,
                n_noise=n_noise,
            )
            fout.write(sample.model_dump_json() + "\n")

    print(f"\nDataset saved to {out_path}")
    print(f"  Samples: {args.n_samples}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="outputs/raw")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--n_noise", default="20")
    p.add_argument("--output", default="outputs/datasets/benchmark_v1.jsonl")
    main(p.parse_args())