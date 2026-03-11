import re
from src.generation.prompts import CLUE_GENERATION_PROMPT


def generate_clues(engine, secret_label: str, n_clues: int) -> list[str]:
    prompt = CLUE_GENERATION_PROMPT.format(secret_label=secret_label, n_clues=n_clues)
    raw = engine.generate(prompt, max_tokens=512, temperature=0.7)[0]
    clues = []
    for line in raw.split("\n"):
        m = re.match(r"^\d+\.\s+(.+)", line.strip())
        if m:
            clues.append(m.group(1).strip())
    return clues[:n_clues]
