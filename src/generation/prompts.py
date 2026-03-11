"""All prompt templates for the benchmark generation pipeline."""

CLUE_GENERATION_PROMPT = """You are a creative writer designing a mystery puzzle.

Secret topic: {secret_label}

Generate exactly {n_clues} indirect clues that together reveal the secret, but each clue alone is ambiguous.
Rules:
- Each clue must be a SHORT factual statement (1-2 sentences) that could appear naturally in a casual email.
- NO clue should directly name or clearly state the secret.
- Together, the clues must logically point to ONLY one conclusion: the secret.
- Number each clue.

Output format:
1. <clue text>
2. <clue text>
..."""

SECRET_DIALOGUE_PROMPT = """You are writing a realistic email exchange between two friends/colleagues.

Participants: {person_a} and {person_b}
Clue to embed (do NOT state it explicitly — weave it in naturally): {clue}
Number of email turns: {n_turns}
Style: casual, natural, like real people writing to each other

Write {n_turns} emails alternating between {person_a} and {person_b}.
Each email must have: Subject, and Body.
The clue should appear subtly — as a side remark, a vague reference, or an indirect hint.

Output format:
---
From: <name>
To: <name>
Subject: <subject>
<body>
---
(repeat for each turn)"""

NOISE_DIALOGUE_PROMPT = """You are writing a realistic email exchange between two friends/colleagues.

Participants: {person_a} and {person_b}
Topic: {topic}
Number of email turns: {n_turns}
Style: casual, natural, unrelated to any secret or serious matter

Write {n_turns} emails alternating between {person_a} and {person_b}.

Output format:
---
From: <name>
To: <name>
Subject: <subject>
<body>
---
(repeat for each turn)"""

EVALUATION_PROMPT = """Below is a long email thread between {person_a} and {person_b}, covering many topics.

Your task: Read ALL the emails carefully and determine if there is a hidden secret that {person_a} or {person_b} is concealing. If yes, describe the secret in one sentence.

Email Thread:
{email_thread}

Answer in this format:
Secret found: <Yes/No>
Secret description: <one sentence, or "None">"""
