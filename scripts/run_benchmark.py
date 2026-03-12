"""
Usage:
  python scripts/run_benchmark.py --model Qwen/Qwen1.5-32B-Chat --dataset outputs/datasets/benchmark_v1.jsonl
"""
import argparse
from src.models.vllm_engine import VLLMEngine
from src.benchmark.evaluator import evaluate_dataset
from src.benchmark.metrics import compute_metrics


def main(args):
    engine = VLLMEngine(model_name=args.model, tensor_parallel_size=args.tp)
    evaluate_dataset(engine, args.dataset, args.output)
    compute_metrics(args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="outputs/results/run1.json")
    main(p.parse_args())
