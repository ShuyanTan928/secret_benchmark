"""
End-to-end dataset generation pipeline.
Usage:
  python scripts/generate_dataset.py --model Qwen/Qwen3.5-35B-A3B --n_samples 50 --n_noise 8-15
"""
import argparse, random
from pathlib import Path
from tqdm import tqdm

from src.models.vllm_engine import VLLMEngine
from src.data_utils.person_sampler import sample_pair
from src.data_utils.topic_sampler import sample_secret, sample_noise_topics
from src.generation.dialogue_generator import generate_secret_dialogues
from src.generation.noise_generator import generate_noise_dialogue
from src.generation.combiner import combine


def main(args):
    engine = VLLMEngine(model_name=args.model, tensor_parallel_size=args.tp)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for i in tqdm(range(args.n_samples), desc="Generating samples"):
            person_a, person_b = sample_pair()
            secret = sample_secret()
            n_noise = random.randint(*[int(x) for x in args.n_noise.split("-")])

            clue_dialogues = generate_secret_dialogues(
                engine, person_a, person_b, secret["clue_breakdown"],
            )
            noise_dialogues = [
                generate_noise_dialogue(engine, person_a, person_b, t)
                for t in sample_noise_topics(n_noise)
            ]
            sample = combine(
                person_a, person_b,
                secret["id"], secret["label"],
                clue_dialogues, noise_dialogues,
            )

            fout.write(sample.model_dump_json() + "\n")
            print(f"  [{i+1}] {person_a} <> {person_b} | secret={secret['id']} | clues={len(clue_dialogues)} | SNR={sample.snr}")

    print(f"\nDataset saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--n_noise", default="8-15")
    p.add_argument("--output", default="outputs/datasets/benchmark_v1.jsonl")
    main(p.parse_args())