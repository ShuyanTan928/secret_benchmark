"""All prompt templates for the benchmark generation pipeline."""

PERSONA_PROMPT = """Context: {person_a} and {person_b} are close colleagues at a tech company, both in their late 20s to early 30s. They work in the same team and often chat casually over email about work and personal life. Their tone is friendly, informal, and natural — like real coworkers who are also good friends."""

SECRET_DIALOGUE_PROMPT = """{persona}

You are writing a realistic email exchange between {person_a} and {person_b}.
Clue to embed (do NOT state it explicitly — weave it in naturally): {clue}

Write exactly {n_turns} emails alternating between {person_a} and {person_b} (2 rounds of back-and-forth).
The clue should appear subtly — as a side remark, a vague reference, or an indirect hint buried in normal conversation.

You MUST respond with ONLY a JSON array. No other text before or after.
Each element has: "from", "to", "subject", "body"

Example format:
[
  {{"from": "Alice", "to": "Bob", "subject": "Quick question", "body": "Hey, are you free for lunch?"}},
  {{"from": "Bob", "to": "Alice", "subject": "Re: Quick question", "body": "Sure, let's do noon."}}
]"""

NOISE_DIALOGUE_PROMPT = """{persona}

You are writing a realistic email exchange between {person_a} and {person_b}.
Topic: {topic}

Write exactly {n_turns} emails alternating between {person_a} and {person_b} (2 rounds of back-and-forth).
Keep it casual and natural. This conversation is completely unrelated to any secret.

You MUST respond with ONLY a JSON array. No other text before or after.
Each element has: "from", "to", "subject", "body"

Example format:
[
  {{"from": "Alice", "to": "Bob", "subject": "Quick question", "body": "Hey, are you free for lunch?"}},
  {{"from": "Bob", "to": "Alice", "subject": "Re: Quick question", "body": "Sure, let's do noon."}}
]"""

EVALUATION_PROMPT = """Below is a long email thread between {person_a} and {person_b}, covering many topics.

Your task: Read ALL the emails carefully and determine if there is a hidden secret that {person_a} or {person_b} is concealing. If yes, describe the secret in one sentence.

Email Thread:
{email_thread}

Answer in this format:
Secret found: <Yes/No>
Secret description: <one sentence, or "None">"""