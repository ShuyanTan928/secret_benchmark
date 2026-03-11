import uuid
from src.generation.prompts import SECRET_DIALOGUE_PROMPT
from src.data_utils.schema import EmailDialogue, EmailTurn


def _parse_emails(raw: str, person_a: str, person_b: str) -> list[EmailTurn]:
    turns = []
    for block in [b.strip() for b in raw.split("---") if b.strip()]:
        lines = block.split("\n")
        sender, recipient, subject, body_lines = "", "", "", []
        for line in lines:
            if line.startswith("From:"):       sender = line.replace("From:", "").strip()
            elif line.startswith("To:"):       recipient = line.replace("To:", "").strip()
            elif line.startswith("Subject:"): subject = line.replace("Subject:", "").strip()
            else:                              body_lines.append(line)
        if sender and subject:
            turns.append(EmailTurn(
                sender=sender or person_a,
                recipient=recipient or person_b,
                subject=subject,
                body="\n".join(body_lines).strip(),
            ))
    return turns


def generate_secret_dialogue(
    engine, person_a: str, person_b: str, clue: str, clue_index: int, n_turns: int = 3
) -> EmailDialogue:
    prompt = SECRET_DIALOGUE_PROMPT.format(
        person_a=person_a, person_b=person_b, clue=clue, n_turns=n_turns)
    raw = engine.generate(prompt, max_tokens=800, temperature=0.85)[0]
    return EmailDialogue(
        dialogue_id=str(uuid.uuid4())[:8],
        topic=f"[SECRET_CLUE_{clue_index}]",
        is_secret_clue=True,
        clue_index=clue_index,
        emails=_parse_emails(raw, person_a, person_b),
    )
