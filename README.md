# Email Benchmark

A benchmark for testing LLM ability to detect hidden secrets in long email chains.

## Setup
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Generate Dataset
```bash
python scripts/generate_dataset.py --model Qwen/Qwen3.5-35B-A3B --n_samples 50
```

## Run Benchmark
```bash
python scripts/run_benchmark.py --model Qwen/Qwen3.5-35B-A3B --dataset outputs/datasets/benchmark_v1.jsonl
```

## Use Llama instead
```bash
python scripts/generate_dataset.py --model meta-llama/Llama-3.3-70B-Instruct --tp 2 --n_samples 50
```

## Pipeline
1. **Clue Dialogues** — For each secret in `data/secrets.json`, generate email dialogues that subtly embed each clue from `clue_breakdown`
2. **Noise Dialogues** — Generate unrelated email conversations from topics in `data/noise_topics.json`
3. **Combine** — Clue dialogues inserted at random positions among noise; difficulty controlled by SNR (clues / total dialogues)

## Configuration
- **Secrets**: Edit `data/secrets.json` to add/remove/modify secret topics and their clue breakdowns. No code changes needed.
- **Noise topics**: Edit `data/noise_topics.json` to change conversation topics.
- **Names**: Edit `data/names.json` to change participant names.
- **Model settings**: Edit `configs/generation_config.yaml` for model and generation parameters.