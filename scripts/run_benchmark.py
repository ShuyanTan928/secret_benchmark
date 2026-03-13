"""
Run the multi-step evaluation benchmark.

Usage:
  python scripts/run_benchmark.py --tester Qwen/Qwen3-14B --dataset outputs/datasets/benchmark_v1.jsonl
  python scripts/run_benchmark.py --tester Qwen/Qwen3-14B --judge gpt-4o-mini --judge_api --dataset outputs/datasets/benchmark_v1.jsonl
"""
import argparse
from src.models.vllm_engine import VLLMEngine
from src.models.api_engine import APIEngine
from src.benchmark.evaluator import evaluate_dataset
from src.benchmark.metrics import compute_metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tester", default="Qwen/Qwen3-14B")
    p.add_argument("--judge", default=None)
    p.add_argument("--judge_api", action="store_true")
    p.add_argument("--api_key", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="outputs/results/run1.json")
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

    evaluate_dataset(tester_engine, judge_engine, args.dataset, args.output)
    compute_metrics(args.output)
