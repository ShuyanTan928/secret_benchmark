# Email Benchmark

A benchmark for testing LLM ability to detect hidden secrets in long email chains.

## Setup
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Project Structure

```
configs/
  benchmark_config.yaml      # Benchmark parameters (n_samples, noise range)
  generation_config.yaml     # Model and generation settings

data/
  names.json                 # Name pool (male/female) for random assignment
  noise_topics.json          # List of mundane email topics
  secrets.json               # Secret definitions and clue breakdowns

scripts/
  generate_raw.py            # Step 1: Generate raw secret + noise dialogues
  assemble_dataset.py        # Step 2: Combine raw dialogues into benchmark
  run_benchmark.py           # Step 3: Evaluate a model on the benchmark
  check_models.py            # Utility: query HuggingFace for available models

src/
  models/
    vllm_engine.py           # Local vLLM inference wrapper
    api_engine.py            # API-based inference (OpenAI, Claude, etc.)
  data_utils/
    schema.py                # Pydantic data models (EmailTurn, BenchmarkSample)
    person_sampler.py         # Random gendered name sampling
    topic_sampler.py          # Load secrets and noise topics from JSON
  generation/
    prompts.py               # All prompt templates
    dialogue_generator.py    # Generate secret clue dialogues
    noise_generator.py       # Generate noise dialogues
    combiner.py              # Combine clues + noise, replace names
  benchmark/
    eval_prompts.py          # Evaluation prompt templates (5-step pipeline)
    evaluator.py             # Multi-step evaluation logic and scoring
    metrics.py               # Compute and display benchmark metrics

outputs/
  raw/                       # Raw generated dialogues (before assembly)
  datasets/                  # Assembled benchmark datasets
  results/                   # Evaluation results
```

## Pipeline

### Step 1: Generate raw dialogues
```bash
# Generate both secret and noise dialogues
python scripts/generate_raw.py --model Qwen/Qwen3-14B --n_per_topic 3

# Generate only secret clue dialogues
python scripts/generate_raw.py --model Qwen/Qwen3-14B --only_secrets

# Generate only noise dialogues
python scripts/generate_raw.py --model Qwen/Qwen3-14B --only_noise --n_per_topic 5
```

| Argument | Description | Default |
|---|---|---|
| `--model` | HuggingFace model for generation | `Qwen/Qwen3-14B` |
| `--tp` | Tensor parallel size (number of GPUs) | `1` |
| `--n_per_topic` | Noise dialogues generated per topic | `3` |
| `--only_secrets` | Only generate secret clue dialogues | `false` |
| `--only_noise` | Only generate noise dialogues | `false` |
| `--output_dir` | Directory for raw output files | `outputs/raw` |

### Step 2: Assemble dataset
```bash
# Fixed noise count
python scripts/assemble_dataset.py --n_samples 50 --n_noise 15

# Random noise count per sample
python scripts/assemble_dataset.py --n_samples 50 --n_noise 8-15
```

| Argument | Description | Default |
|---|---|---|
| `--n_samples` | Number of benchmark samples to create | `50` |
| `--n_noise` | Noise dialogues per sample (fixed or range) | `8-15` |
| `--raw_dir` | Directory with raw dialogues | `outputs/raw` |
| `--output` | Output benchmark file | `outputs/datasets/benchmark_v1.jsonl` |

### Step 3: Run evaluation
```bash
# Same model as tester and judge
python scripts/run_benchmark.py \
  --tester Qwen/Qwen3-14B \
  --dataset outputs/datasets/benchmark_v1.jsonl

# Local tester + API-based judge
export OPENAI_API_KEY=your_key_here
python scripts/run_benchmark.py \
  --tester Qwen/Qwen3-14B \
  --judge gpt-4o-mini \
  --judge_api \
  --dataset outputs/datasets/benchmark_v1.jsonl

# Local tester + Claude as judge
export OPENAI_API_KEY=your_anthropic_key
export OPENAI_BASE_URL=https://api.anthropic.com/v1
python scripts/run_benchmark.py \
  --tester Qwen/Qwen3-14B \
  --judge claude-sonnet-4-20250514 \
  --judge_api \
  --dataset outputs/datasets/benchmark_v1.jsonl
```

| Argument | Description | Default |
|---|---|---|
| `--tester` | Local model being evaluated | `Qwen/Qwen3-14B` |
| `--judge` | Judge model name | Same as tester |
| `--judge_api` | Use API for judge instead of local vLLM | `false` |
| `--api_key` | API key (or set `OPENAI_API_KEY` env var) | `None` |
| `--base_url` | API base URL (or set `OPENAI_BASE_URL` env var) | `None` |
| `--tp` | Tensor parallel size | `1` |
| `--dataset` | Path to benchmark dataset | Required |
| `--output` | Path to save results | `outputs/results/run1.json` |

## Scoring

| Score | Meaning |
|---|---|
| 0 | Secret not detected |
| 1 | Detected but wrong identification |
| 2 | Correct identification but no valid evidence cited |
| 3 | Correct identification, some valid evidence but also cited noise |
| 4 | Correct identification, all cited evidence is valid but incomplete |
| 5 | Fully correct: right secret, all clues found, no noise cited |

Each sample also reports **Precision** (correct citations / total citations) and **Recall** (correct citations / total clues).

## Configuration

- **Secrets**: Edit `data/secrets.json` to add/remove/modify secret topics and clue breakdowns. No code changes needed.
- **Noise topics**: Edit `data/noise_topics.json` to change conversation topics.
- **Names**: Edit `data/names.json` to change participant name pools (male/female).
- **Model settings**: Edit `configs/generation_config.yaml` for model and generation parameters.