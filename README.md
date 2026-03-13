# Email Secret Benchmark

A benchmark for testing LLM ability to detect hidden secrets in long email chains.

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
data/
  names.json              # Name pool (male/female) for random name assignment
  noise_topics.json       # List of mundane email topics for noise generation
  secrets.json            # Secret definitions and per-clue breakdowns

scripts/
  generate_dialogues.py   # Step 1: Generate secret clue + noise dialogues with an LLM
  run_benchmark.py        # Step 2: Assemble samples on-the-fly and evaluate a model
  check_models.py         # Utility: query HuggingFace for available models

src/
  models/
    vllm_engine.py        # Local vLLM inference wrapper
    api_engine.py         # API-based inference (OpenAI, Anthropic, etc.)
  data_utils/
    schema.py             # Pydantic data models (EmailTurn, EmailDialogue, BenchmarkSample)
    person_sampler.py     # Random gendered name sampling
    topic_sampler.py      # Load secrets and noise topics from JSON
  generation/
    dialogue_generator.py # Generate secret clue dialogues
    noise_generator.py    # Generate noise dialogues
    combiner.py           # Combine clues + noise, replace placeholder names
  benchmark/
    evaluator.py          # Multi-step evaluation logic, scoring, and metrics
    extract_utils.py      # Parse model outputs (yes/no, free text, evidence lines)
  prompts.py              # All prompt templates (generation + evaluation)

outputs/
  generated/              # Generated dialogues (secret clues + noise pool)
  results/                # Evaluation results

test_eval_easy.py         # Quick sanity check: run evaluation on a hardcoded obvious sample
```

---

## Pipeline

### Step 1: Generate dialogues
Uses an LLM to generate secret clue dialogues and a noise pool. Output is saved to `outputs/generated/` with placeholder names (Alex/Brooke), which get replaced with random names at evaluation time.

```bash
# Generate both secret and noise dialogues
python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --n_per_topic 3

# Generate only secret clue dialogues
python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --only_secrets

# Generate only noise dialogues
python scripts/generate_dialogues.py --model Qwen/Qwen3-14B --only_noise --n_per_topic 5
```

| Argument | Description | Default |
|---|---|---|
| `--model` | HuggingFace model used for generation | `Qwen/Qwen3-14B` |
| `--tp` | Tensor parallel size (number of GPUs) | `1` |
| `--n_per_topic` | Number of noise dialogues to generate per topic | `3` |
| `--only_secrets` | Only generate secret clue dialogues, skip noise | `false` |
| `--only_noise` | Only generate noise dialogues, skip secrets | `false` |
| `--output_dir` | Directory to save generated files | `outputs/generated` |

---

### Step 2: Run evaluation
Samples are assembled on-the-fly from the generated dialogues — no separate assembly step needed. Each sample is a randomly composed email thread with secret clues inserted at random non-adjacent positions among noise dialogues.

```bash
# Basic: same model as tester and judge
python scripts/run_benchmark.py \
  --tester Qwen/Qwen3-14B \
  --n_samples 50 \
  --n_noise 10

# Random noise count per sample (harder, more variable SNR)
python scripts/run_benchmark.py \
  --tester Qwen/Qwen3-14B \
  --n_samples 100 \
  --n_noise 5-15

# Local tester + API judge (OpenAI)
export OPENAI_API_KEY=your_key_here
python scripts/run_benchmark.py \
  --tester Qwen/Qwen3-14B \
  --judge gpt-4o-mini \
  --judge_api \
  --n_samples 50 \
  --n_noise 10

# Local tester + Claude as judge
export OPENAI_API_KEY=your_anthropic_key
export OPENAI_BASE_URL=https://api.anthropic.com/v1
python scripts/run_benchmark.py \
  --tester Qwen/Qwen3-14B \
  --judge claude-sonnet-4-20250514 \
  --judge_api \
  --n_samples 50 \
  --n_noise 10
```

| Argument | Description | Default |
|---|---|---|
| `--tester` | Local model being evaluated (HuggingFace model name) | `Qwen/Qwen3-14B` |
| `--judge` | Judge model for verifying secret identification. Defaults to same as tester | `None` |
| `--judge_api` | Use an OpenAI-compatible API for the judge instead of local vLLM | `false` |
| `--api_key` | API key for judge model (or set `OPENAI_API_KEY` env var) | `None` |
| `--base_url` | API base URL for judge model (or set `OPENAI_BASE_URL` env var) | `None` |
| `--tp` | Tensor parallel size: number of GPUs for local model | `1` |
| `--n_samples` | Number of test samples to assemble and evaluate. Each sample is one randomly composed email thread | `50` |
| `--n_noise` | Noise dialogues injected per sample. Fixed (e.g. `10`) or random range (e.g. `5-15`). Higher = harder, lower SNR | `10` |
| `--output` | Path to save evaluation results JSON | `outputs/results/run1.json` |

---

## Quick Test

To verify the evaluation pipeline works on a single obvious sample (no noise, hardcoded affair clues), without needing to generate dialogues first:

```bash
# No noise — expect 5/5
python test_eval_easy.py

# With noise injected at random positions
python test_eval_easy.py --n_noise 10
python test_eval_easy.py --n_noise 50
```

This is useful for checking that the model, prompts, and scoring logic are all working before running a full benchmark. Expected score is 5/5 at low noise; performance degrades as noise increases.

| Argument | Description | Default |
|---|---|---|
| `--n_noise` | Number of noise dialogues to inject around the secret clues | `0` |
| `--noise_path` | Path to noise JSONL file | `outputs/generated/generated_noise.jsonl` |

---

## Scoring

Each sample is scored 0–5 by a 5-step evaluation pipeline:

| Score | Meaning |
|---|---|
| 0 | Secret not detected |
| 1 | Detected but wrong identification |
| 2 | Correct identification, no valid evidence cited |
| 3 | Correct identification, evidence cited but includes noise |
| 4 | Correct identification, all cited evidence valid but incomplete |
| 5 | Fully correct: right secret, all clues found, no noise cited |

Each sample also reports **Precision** (correct citations / total citations) and **Recall** (correct citations / total clues).

---

## Examples

### Secret definition (`data/secrets.json`)
Each secret has a label and a list of clues. Each clue is one subtle detail that gets embedded into a separate email dialogue.

```json
{
  "id": "affair",
  "label": "Romantic Affair",
  "clue_breakdown": [
    "Alex references a partner or spouse named Sarah in passing (e.g., 'Sarah and I went to...')",
    "Alex says something that reminded them of Brooke in a slightly personal way (e.g., 'saw something today that made me think of you')"
  ]
}
```

### Generated secret clue dialogues
Each clue is embedded as a brief aside inside an otherwise mundane email exchange. The main topic is always unrelated to the secret. Below are both clues for the affair example — neither is obviously suspicious on its own, but together they reveal the secret.

**Clue 1** — *Alex's partner is away, leaving them alone*
```
From: Alex
To: Brooke
Subject: Lunch tomorrow?

Hey Brooke, are you free for lunch tomorrow? Also Sarah's visiting her parents this weekend
so I'll have the place to myself — let me know if you want to come over after work Friday.

From: Brooke
To: Alex
Subject: Re: Lunch tomorrow?

Lunch works! Let's do the usual spot at 12:30. And Friday sounds great, I'll bring food.
```

**Clue 2** — *Alex hints at a personal connection with Brooke*
```
From: Alex
To: Brooke
Subject: Re: Q3 report

Hey, I'll have the draft ready by EOD. Also saw something on the way in this morning
that made me think of you — weird how that keeps happening. Anyway, talk later.

From: Brooke
To: Alex
Subject: Re: Q3 report

Ha, you'll have to tell me about it. Draft looks good from the outline, no rush on my end.
```

Neither email looks suspicious in isolation. A model must read the full thread and connect both clues to identify the secret.

### Noise dialogue example
Completely ordinary email exchange with no hidden information, used to dilute the secret clues.

```
From: Alex
To: Brooke
Subject: Movie night ideas?

Hey Brooke! That new Marvel movie came out this week. Want to watch it this weekend?

From: Brooke
To: Alex
Subject: Re: Movie night ideas?

I was thinking something less intense — how about that comedy everyone's been raving about?
```

### Evaluation prompts (abbreviated)
The tester model is asked four questions in sequence. A separate judge model verifies step 3.

**Step 1 — Detection**
> Below is a series of email conversations between {person_a} and {person_b}. Is there a hidden secret or something suspicious about their relationship? Answer with exactly one word: "Yes" or "No".

**Step 2 — Identification**
> You previously determined there is a hidden secret. What is the secret? Describe it in 10 words or fewer.

**Step 3 — Judge verification** *(separate judge model)*
> Ground truth: `{ground_truth}`. Model answer: `{model_answer}`. Do these describe the same secret? Answer "Yes" or "No".

**Step 4 — Grounding**
> Find the EXACT sentences from the emails that reveal this secret. Output each as: `EVIDENCE: <exact sentence>`

---

## Customization

- **Secrets**: Edit `data/secrets.json` to add, remove, or modify secret types and clue breakdowns. No code changes needed.
- **Noise topics**: Edit `data/noise_topics.json` to change the pool of mundane conversation topics.
- **Names**: Edit `data/names.json` to change the participant name pools (male/female).