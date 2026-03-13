import uuid
from src.prompts import NOISE_DIALOGUE_PROMPT, PERSONA_PROMPT
from src.generation.dialogue_generator import _parse_emails_json
from src.data_utils.schema import EmailDialogue


def generate_noise_dialogue(engine, topic: str, n_turns: int = 4) -> EmailDialogue:
    """Generate one noise dialogue. Uses PERSON_A/PERSON_B placeholders."""
    persona = PERSONA_PROMPT
    prompt = NOISE_DIALOGUE_PROMPT.format(
        persona=persona, topic=topic, n_turns=n_turns,
    )
    raw = engine.generate(prompt, max_tokens=1024, temperature=0.9)[0]
    emails = _parse_emails_json(raw)

    # Retry once if parsing failed
    if len(emails) == 0:
        raw = engine.generate(prompt, max_tokens=1024, temperature=0.9)[0]
        emails = _parse_emails_json(raw)
        emails = _parse_emails_json(raw)

    return EmailDialogue(
        dialogue_id=str(uuid.uuid4())[:8],
        topic=topic,
        is_secret_clue=False,
        emails=emails,
    )


def generate_noise_pool(engine, topics: list[str], n_per_topic: int = 3) -> list[EmailDialogue]:
    """Generate a pool of noise dialogues. Multiple per topic for variety."""
    pool = []
    for topic in topics:
        for _ in range(n_per_topic):
            pool.append(generate_noise_dialogue(engine, topic))
    return pool