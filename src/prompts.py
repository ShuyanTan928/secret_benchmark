"""
All prompt templates for the benchmark pipeline.
Covers both generation (dialogue creation) and evaluation (multi-step scoring).
"""

# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------

# Default placeholder names used during generation.
# These get replaced with random names during the assembly step.
PLACEHOLDER_A = "Alex"
PLACEHOLDER_B = "Brooke"

PERSONA_PROMPT = f"""Context: {PLACEHOLDER_A} and {PLACEHOLDER_B} are close colleagues at a tech company, both in their late 20s to early 30s. They work in the same team and often chat casually over email about work and personal life. Their tone is friendly, informal, and natural — like real coworkers who are also good friends."""

SECRET_DIALOGUE_PROMPT = """/no_think
{persona}

Write a short, realistic email exchange between Alex and Brooke (2 emails total: one message and one reply).

The email should be about an everyday work or life topic (e.g., a meeting, lunch, a deadline, weekend plans — pick one naturally). Somewhere in the conversation, one of them casually mentions the following background detail as a BRIEF side comment — not as the main topic:

Background detail to include: {clue}

CRITICAL RULES:
- The background detail must appear as a single short phrase or clause buried inside a longer sentence about something else. It should feel like an afterthought.
- The main topic of the email must be something COMPLETELY DIFFERENT from the background detail.
- Do NOT draw any attention to the background detail. No follow-up questions about it, no reactions to it, no elaboration.
- The reply should respond to the MAIN topic only, not to the background detail.

BAD example (too obvious):
  "Hey, I approved your promotion request today! Congrats!"
GOOD example (subtle):
  "Hey, can you send me the slide deck before 3? Also I submitted that paperwork you asked about, should be all set."

You MUST respond with ONLY a JSON array. No other text before or after.

[
  {{"from": "Alex", "to": "Brooke", "subject": "...", "body": "..."}},
  {{"from": "Brooke", "to": "Alex", "subject": "...", "body": "..."}}
]"""

NOISE_DIALOGUE_PROMPT = """/no_think
{persona}

Write a realistic email exchange between Alex and Brooke ({n_turns} emails total, alternating between them).

Topic: {topic}

Rules:
- Keep it casual and natural. This is a completely ordinary, mundane conversation.

You MUST respond with ONLY a JSON array. No other text before or after.

[
  {{"from": "Alex", "to": "Brooke", "subject": "...", "body": "..."}},
  {{"from": "Brooke", "to": "Alex", "subject": "...", "body": "..."}}
]"""

# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

# Step 1: Detection
DETECTION_PROMPT = """Below is a series of email conversations between {person_a} and {person_b}.

{email_thread}

Question: Based on these emails, is there a hidden secret, inappropriate behavior, or something suspicious about the relationship between {person_a} and {person_b}?

Rules:
- Answer with EXACTLY one word: "Yes" or "No".
- Do NOT include any explanation, reasoning, or additional text.

Your response:"""

# Step 2: Identification
IDENTIFICATION_PROMPT = """Below is a series of email conversations between {person_a} and {person_b}.

{email_thread}

You previously determined there is a there a hidden secret, inappropriate behavior, or something suspicious in these emails.
What is the secret, inappropriate behavior, or something suspicious? Describe it in 10 words or fewer.

Rules:
- Output ONLY the secret description, nothing else.
- Do NOT repeat the question or add any explanation.
- Maximum 10 words.

Your response:"""

# Step 3: Verification (Judge LLM)
JUDGE_PROMPT = """You are a judge deciding whether a model correctly identified the hidden behavior in a set of emails.

Here is what actually happened in the emails (the ground truth):

Secret category: {ground_truth}

Key evidence from the emails:
{clue_bodies}

Now here is what the model said the behavior was:
Model's answer: {model_answer}

Question: Does the model's answer correctly capture the behavior shown in the evidence above? The model does not need to use the exact same words — it just needs to describe the same underlying behavior or misconduct.

For example, if the evidence shows someone making inappropriate comments and leveraging a performance review, and the model says "workplace flirtation" or "sexual harassment" or "inappropriate behavior by a boss", those would all be correct.

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