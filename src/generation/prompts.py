"""All prompt templates for the benchmark generation pipeline."""

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

EVALUATION_PROMPT = """Below is a long email thread between {person_a} and {person_b}, covering many topics.

Your task: Read ALL the emails carefully and determine if there is a hidden secret that {person_a} or {person_b} is concealing. If yes, describe the secret in one sentence.

Email Thread:
{email_thread}

Answer in this format:
Secret found: <Yes/No>
Secret description: <one sentence, or "None">"""