import uuid
from src.generation.prompts import NOISE_DIALOGUE_PROMPT, PERSONA_PROMPT
from src.generation.dialogue_generator import _parse_emails_json
from src.data_utils.schema import EmailDialogue


def generate_noise_dialogue(
    engine, person_a: str, person_b: str, topic: str, n_turns: int = 4
) -> EmailDialogue:
    persona = PERSONA_PROMPT.format(person_a=person_a, person_b=person_b)
    prompt = NOISE_DIALOGUE_PROMPT.format(
        persona=persona, person_a=person_a, person_b=person_b,
        topic=topic, n_turns=n_turns,
    )
    raw = engine.generate(prompt, max_tokens=1024, temperature=0.9)[0]
    emails = _parse_emails_json(raw, person_a, person_b)

    # Retry once if parsing failed
    if len(emails) == 0:
        raw = engine.generate(prompt, max_tokens=1024, temperature=0.9)[0]
        emails = _parse_emails_json(raw, person_a, person_b)

    return EmailDialogue(
        dialogue_id=str(uuid.uuid4())[:8],
        topic=topic,
        is_secret_clue=False,
        emails=emails,
    )