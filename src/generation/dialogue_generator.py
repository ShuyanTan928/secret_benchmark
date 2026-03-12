import uuid, json, re
from src.generation.prompts import SECRET_DIALOGUE_PROMPT, PERSONA_PROMPT, PLACEHOLDER_A, PLACEHOLDER_B
from src.data_utils.schema import EmailDialogue, EmailTurn


def _parse_emails_json(raw: str) -> list[EmailTurn]:
    """Parse LLM output as JSON array."""
    text = raw.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try to find JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        emails = json.loads(text)
        turns = []
        for e in emails:
            turns.append(EmailTurn(
                sender=e.get("from", PLACEHOLDER_A),
                recipient=e.get("to", PLACEHOLDER_B),
                subject=e.get("subject", ""),
                body=e.get("body", ""),
            ))
        return turns
    except (json.JSONDecodeError, TypeError, KeyError):
        # Fallback: try to extract individual JSON objects
        turns = []
        for obj_match in re.finditer(r"\{[^{}]+\}", text):
            try:
                e = json.loads(obj_match.group(0))
                turns.append(EmailTurn(
                    sender=e.get("from", PLACEHOLDER_A),
                    recipient=e.get("to", PLACEHOLDER_B),
                    subject=e.get("subject", ""),
                    body=e.get("body", ""),
                ))
            except (json.JSONDecodeError, TypeError):
                continue
        return turns


def generate_secret_dialogue(engine, clue: str) -> EmailDialogue:
    """Generate one email dialogue for a single clue. Uses PERSON_A/PERSON_B placeholders."""
    persona = PERSONA_PROMPT
    prompt = SECRET_DIALOGUE_PROMPT.format(persona=persona, clue=clue)
    raw = engine.generate(prompt, max_tokens=512, temperature=0.85, stop=["\n\n[", "\n\n{"])[0]
    emails = _parse_emails_json(raw)

    # Retry once if parsing failed
    if len(emails) == 0:
        raw = engine.generate(prompt, max_tokens=512, temperature=0.85)[0]
        emails = _parse_emails_json(raw)

    # Only keep first 2 emails to enforce one round of back-and-forth
    emails = emails[:2]

    # Only keep first 2 emails
    emails = emails[:2]

    return EmailDialogue(
        dialogue_id=str(uuid.uuid4())[:8],
        topic="[SECRET_CLUE]",
        is_secret_clue=True,
        clue_description=clue,
        emails=emails,
    )


def generate_all_secret_dialogues(engine, secrets: list[dict]) -> dict:
    """Generate one dialogue per clue for all secrets. Returns {secret_id: [dialogues]}."""
    result = {}
    for secret in secrets:
        dialogues = []
        for clue in secret["clue_breakdown"]:
            dlg = generate_secret_dialogue(engine, clue)
            dialogues.append(dlg)
        result[secret["id"]] = dialogues
    return result