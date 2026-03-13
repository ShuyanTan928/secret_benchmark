"""Evaluation prompts for the benchmark pipeline.

Design notes for Qwen3 / thinking-mode models:
  - End every yes/no prompt with an explicit single-word constraint.
  - Use "Your response:" as the final line to anchor generation.
  - For structured output (EVIDENCE lines), give a strict format block
    and explicitly forbid explanations after the evidence.
"""

# Step 1: Detection
DETECTION_PROMPT = """Below is a series of email conversations between {person_a} and {person_b}.

{email_thread}

Question: Based on these emails, is there a hidden secret or something suspicious about the relationship between {person_a} and {person_b}?

Rules:
- Answer with EXACTLY one word: "Yes" or "No".
- Do NOT include any explanation, reasoning, or additional text.

Your response:"""

# Step 2: Identification
IDENTIFICATION_PROMPT = """Below is a series of email conversations between {person_a} and {person_b}.

{email_thread}

You previously determined there is a hidden secret in these emails.
What is the secret? Describe it in 10 words or fewer.

Rules:
- Output ONLY the secret description, nothing else.
- Do NOT repeat the question or add any explanation.
- Maximum 10 words.

Your response:"""

# Step 3: Verification (Judge LLM)
JUDGE_PROMPT = """You are a judge comparing two descriptions of a secret.

Ground truth secret: {ground_truth}
Model's answer: {model_answer}

Do the ground truth and the model's answer describe the same secret? Ignore wording differences — focus on whether the core meaning matches.

Rules:
- Answer with EXACTLY one word: "Yes" or "No".
- Do NOT include any explanation, reasoning, or additional text.

Your response:"""

# Step 4: Grounding
GROUNDING_PROMPT = """Below is a series of email conversations between {person_a} and {person_b}.

{email_thread}

The secret hidden in these emails is: {model_answer}

Your task: find the EXACT sentences from the emails above that reveal this secret. Copy each sentence word-for-word from the original emails. Do not paraphrase, summarize, or add commentary.

Output format — each piece of evidence on its own line prefixed with "EVIDENCE:":
EVIDENCE: <exact sentence from email>
EVIDENCE: <exact sentence from email>

Rules:
- Copy text EXACTLY as it appears in the emails above.
- Each EVIDENCE line must contain one sentence from the emails.
- Output ONLY "EVIDENCE:" lines. No explanations, no numbering, no headers.
- Do NOT output anything after the last EVIDENCE line.

Your response:"""