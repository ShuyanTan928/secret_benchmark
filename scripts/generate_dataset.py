"""
End-to-end dataset generation pipeline.
Usage:
  python scripts/generate_dataset.py --model Qwen/Qwen1.5-32B-Chat --n_samples 50 --n_clues 3-5 --n_noise 8-15
"""
import argparse, json, random
from pathlib import Path
from tqdm import tqdm

from src.models.vllm_engine import VLLMEngine
from src.data_utils.person_sampler import sample_pair
from src.data_utils.topic_sampler import sample_secret, sample_noise_topics
from src.generation.secret_generator import generate_clues
from src.generation.dialogue_generator import generate_secret_dialogue
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
            n_clues = random.randint(*[int(x) for x in args.n_clues.split("-")])
            n_noise = random.randint(*[int(x) for x in args.n_noise.split("-")])

            clues = generate_clues(engine, secret["label"], n_clues)
            clue_dialogues = [generate_secret_dialogue(engine, person_a, person_b, c, idx+1) for idx, c in enumerate(clues)]
            noise_dialogues = [generate_noise_dialogue(engine, person_a, person_b, t) for t in sample_noise_topics(n_noise)]
            sample = combine(person_a, person_b, secret["id"], secret["label"], clue_dialogues, noise_dialogues)

            fout.write(sample.model_dump_json() + "\n")
            print(f"  [{i+1}] {person_a} ↔ {person_b} | secret={secret['id']} | SNR={sample.snr}")

    print(f"\n✅ Dataset saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen1.5-32B-Chat")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--n_clues", default="3-5")
    p.add_argument("--n_noise", default="8-15")
    p.add_argument("--output", default="outputs/datasets/benchmark_v1.jsonl")
    main(p.parse_args())
