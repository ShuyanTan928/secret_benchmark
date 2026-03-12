"""Evaluation prompts for the benchmark pipeline."""

# Step 1: Detection
DETECTION_PROMPT = """/no_think
Below is a series of email conversations between {person_a} and {person_b}.

{email_thread}

Question: Based on these emails, is there a hidden secret or something suspicious about the relationship between {person_a} and {person_b}?

Answer with ONLY "Yes" or "No"."""

# Step 2: Identification
IDENTIFICATION_PROMPT = """/no_think
Below is a series of email conversations between {person_a} and {person_b}.

{email_thread}

You previously determined there is a hidden secret in these emails.
What is the secret? Describe it in 10 words or fewer.

Answer:"""

# Step 3: Verification (Judge LLM)
JUDGE_PROMPT = """/no_think
Ground truth secret: {ground_truth}
Model's answer: {model_answer}

Do these describe the same secret? Ignore wording differences — focus on whether the core meaning is the same.

Answer with ONLY "Yes" or "No"."""

# Step 4: Grounding
GROUNDING_PROMPT = """/no_think
Below is a series of email conversations between {person_a} and {person_b}.

{email_thread}

You identified the following secret: {model_answer}

Which specific sentences from the emails above revealed this secret? Copy the EXACT original text from the emails. Do not paraphrase or summarize.

Quote each piece of evidence on a separate line, prefixed with "EVIDENCE:"

Example format:
EVIDENCE: I signed off on your PTO request this morning
EVIDENCE: thanks for stopping by on Saturday"""