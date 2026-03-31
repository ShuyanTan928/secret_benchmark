# Email Secret Benchmark

A benchmark for testing LLM ability to detect hidden secrets embedded across long email chains.

---

## Setup
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Project Structure

```
secret_benchmark/
├── data/
│   ├── secret_topics.json             # Secret topic list (plain strings, like noise_topics)
│   ├── secret_clue_breakdowns.json    # Structured clue definitions per secret
│   ├── noise_topics.json              # Mundane email topics for noise generation
│   └── names.json                     # Name pool for random assignment
│
├── benchmark_pool/
│   ├── secret_emails.jsonl            # Generated secret clue email chains (placeholder names)
│   └── noise_emails.jsonl             # Generated noise email chains (placeholder names)
│
├── results/                           # Evaluation results (JSONL per model/config)
├── plots/                             # Generated charts and visualizations
│
├── scripts/
│   ├── generate_dialogues.py          # Generate secret + noise dialogues with LLM
│   └── run_benchmark.py               # Assemble samples and evaluate a model
│
├── src/
│   ├── prompts.py                     # All prompt templates (generation + evaluation)
│   ├── models/
│   │   ├── vllm_engine.py             # Local vLLM inference with model presets
│   │   └── api_engine.py              # API-based inference (OpenAI-compatible)
│   ├── benchmark/
│   │   ├── evaluator.py               # Multi-step evaluation, scoring, metrics
│   │   └── extract_utils.py           # Parse model outputs (yes/no, text, evidence)
│   ├── data_utils/
│   │   ├── schema.py                  # Pydantic models (EmailTurn, EmailDialogue, BenchmarkSample)
│   │   ├── person_sampler.py          # Random gendered name sampling
│   │   └── topic_sampler.py           # Load secrets and noise topics from JSON
│   └── generation/
│       ├── dialogue_generator.py      # Generate secret clue dialogues
│       ├── noise_generator.py         # Generate noise dialogues
│       └── combiner.py               # Combine clues + noise, replace placeholder names
│
├── test_eval_easy.py                  # Quick sanity test with hardcoded samples
└── README.md
```

---

## Data Files

### `data/secret_topics.json`
Plain list of secret topic strings (same format as `noise_topics.json`):
```json
[
  "having a romantic affair while married",
  "accepting bribes or kickbacks from a vendor",
  "trading stocks based on insider company information",
  "workplace harassment or abuse of power over a subordinate"
]
```

### `data/secret_clue_breakdowns.json`
Structured definitions with clue descriptions that get embedded into emails:
```json
[
  {
    "id": "affair",
    "label": "Romantic Affair",
    "clue_breakdown": [
      "Alex references a partner or spouse named Sarah in passing ...",
      "Alex says something that reminded them of Brooke ..."
    ]
  }
]
```

### `data/noise_topics.json`
```json
[
  "discussing a new restaurant opening nearby",
  "talking about a streaming service they just subscribed to",
  "planning a camping trip for next month"
]
```

---

## Pipeline

### Step 1: Generate Dialogues

Uses an LLM to generate secret clue emails and a noise pool. Emails use placeholder names (Alex/Brooke) which get replaced with random names at evaluation time.

```bash
python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --n_per_topic 3
python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --only_secrets
python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --only_noise --n_per_topic 5
```

| Argument | Description | Default |
|---|---|---|
| `--model` | HuggingFace model for generation | `Qwen/Qwen3-14B` |
| `--tp` | Tensor parallel size (GPUs) | `1` |
| `--n_per_topic` | Noise dialogues per topic | `3` |
| `--only_secrets` | Only generate secret clue dialogues | `false` |
| `--only_noise` | Only generate noise dialogues | `false` |
| `--output_dir` | Output directory | `benchmark_pool` |

### Step 2: Run Benchmark

Assembles samples on-the-fly (combining clues + noise, replacing names) and evaluates a model.

```bash
python scripts/run_benchmark.py --tester qwen3-14b --dataset benchmark_pool
```

### Step 3: Quick Sanity Test

`test_eval_easy.py` uses hardcoded, obvious secret samples to quickly verify the evaluation pipeline works. No dataset generation needed.

**Test mode** — run a model on easy samples, save raw outputs:
```bash
python test_eval_easy.py test \
    --model qwen3-14b \
    --secret all \
    --n_noise 0,10,20,50,100 \
    --n_runs 10
```

**Evaluate mode** — score saved results with a judge model:
```bash
# Local judge (same model)
python test_eval_easy.py evaluate \
    --input results/qwen3-14b_n0_10_20_50_100_r10_20260331.jsonl

# API-based judge
python test_eval_easy.py evaluate \
    --input results/qwen3-14b_n0_10_20_50_100_r10_20260331.jsonl \
    --judge gemma-3-27b-it \
    --judge_api \
    --api_key YOUR_KEY \
    --base_url https://generativelanguage.googleapis.com/v1beta/openai/
```

| Argument (test) | Description | Default |
|---|---|---|
| `--model` | Tester model preset | `qwen3-14b` |
| `--secret` | `affair`, `bribery`, `insider_trading`, `harassment`, or `all` | `affair` |
| `--n_noise` | Comma-separated noise levels | `0` |
| `--n_runs` | Runs per secret+noise combination | `10` |
| `--output` | JSONL output path (auto-generated if omitted) | auto |

| Argument (evaluate) | Description | Default |
|---|---|---|
| `--input` | JSONL file from test mode | required |
| `--judge` | Judge model name | same as tester |
| `--judge_api` | Use API for judge | `false` |
| `--api_key` | API key | env `OPENAI_API_KEY` |
| `--base_url` | API base URL | env `OPENAI_BASE_URL` |

---

## Scoring

| Score | Meaning |
|---|---|
| 0 | Secret not detected |
| 1 | Detected but wrong identification |
| 2 | Correct identification but no valid evidence cited |
| 3 | Correct identification, some valid evidence but also cited noise |
| 4 | Correct identification, all cited evidence valid but incomplete |
| 5 | Fully correct: right secret, all clues found, no noise cited |

Each sample also reports **Precision** (correct citations / total citations) and **Recall** (correct citations / total clues).

---

## Sample Content

### Secret Clue Email (affair, clue 1)
```
From: Alex    To: Brooke
Subject: Quick sync before standup?

Hey Brooke, can you send me the slide deck before 3? Sarah and I
went to that new café downtown this weekend — the one near the park.
Anyway, let me know about the slides.
```

### Noise Email (topic: cooking)
```
From: Alex    To: Brooke
Subject: That pasta recipe

Hey, did you ever try making that pesto from scratch? I attempted
it last weekend and it turned out surprisingly decent.
```

### Evaluation Flow
```
Step 1 (Detection):      "Is there a hidden secret?" → Yes/No
Step 2 (Identification): "What is the secret?" → free text (≤10 words)
Step 3 (Judge):          "Does the answer match ground truth?" → Yes/No
Step 4 (Grounding):      "Find exact evidence sentences" → EVIDENCE: ...
Step 5 (Scoring):        Fuzzy-match evidence against clue texts → 0-5
```