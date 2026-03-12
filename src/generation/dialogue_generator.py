import uuid, json, re
from src.generation.prompts import SECRET_DIALOGUE_PROMPT, PERSONA_PROMPT
from src.data_utils.schema import EmailDialogue, EmailTurn


def _parse_emails_json(raw: str, person_a: str, person_b: str) -> list[EmailTurn]:
    """Parse LLM output as JSON array. Falls back to regex extraction if JSON fails."""
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
                sender=e.get("from", person_a),
                recipient=e.get("to", person_b),
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
                    sender=e.get("from", person_a),
                    recipient=e.get("to", person_b),
                    subject=e.get("subject", ""),
                    body=e.get("body", ""),
                ))
            except (json.JSONDecodeError, TypeError):
                continue
        return turns


def generate_secret_dialogues(
    engine, person_a: str, person_b: str,
    clue_breakdown: list[str], n_turns: int = 4,
) -> list[EmailDialogue]:
    """Generate one email dialogue per clue in clue_breakdown."""
    dialogues = []
    for idx, clue in enumerate(clue_breakdown):
        persona = PERSONA_PROMPT.format(person_a=person_a, person_b=person_b)
        prompt = SECRET_DIALOGUE_PROMPT.format(
            persona=persona, person_a=person_a, person_b=person_b,
            clue=clue, n_turns=n_turns,
        )
        raw = engine.generate(prompt, max_tokens=1024, temperature=0.85)[0]
        emails = _parse_emails_json(raw, person_a, person_b)

        # Retry once if parsing failed
        if len(emails) == 0:
            raw = engine.generate(prompt, max_tokens=1024, temperature=0.85)[0]
            emails = _parse_emails_json(raw, person_a, person_b)

        dialogues.append(EmailDialogue(
            dialogue_id=str(uuid.uuid4())[:8],
            topic=f"[SECRET_CLUE_{idx + 1}]",
            is_secret_clue=True,
            clue_index=idx + 1,
            clue_description=clue,
            emails=emails,
        ))
    return dialogues