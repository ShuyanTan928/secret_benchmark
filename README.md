# Email Benchmark

A benchmark for testing LLM ability to detect hidden secrets in long email chains.

## Setup
```bash
pip install -r requirements.txt
```

## Generate Dataset
```bash
python scripts/generate_dataset.py --model Qwen/Qwen1.5-32B-Chat --n_samples 50
```

## Run Benchmark
```bash
python scripts/run_benchmark.py --dataset outputs/datasets/benchmark_v1.jsonl
```

## Check Latest Models
```bash
python scripts/check_models.py --top_k 20
```

## Pipeline
1. **Secret Gen** — LLM generates indirect clues for a secret topic
2. **Noise Gen** — LLM generates unrelated conversations
3. **Combine** — Clues inserted at random positions; SNR = clues / (clues + noise)
