import json, random
from pathlib import Path

SECRET_TOPICS_PATH = Path("data/secret_topics.json")
CLUE_BREAKDOWNS_PATH = Path("data/secret_clue_breakdowns.json")
NOISE_PATH = Path("data/noise_topics.json")

DEFAULT_NOISE = [
    "movies", "sports", "cooking", "travel", "music",
    "weekend plans", "pets", "fitness", "tech gadgets", "books",
]


def load_secret_topics() -> list[dict]:
    """Load secret topic list (id, label, n_clues). No clue details."""
    if not SECRET_TOPICS_PATH.exists():
        raise FileNotFoundError(f"{SECRET_TOPICS_PATH} not found.")
    return json.loads(SECRET_TOPICS_PATH.read_text())


def load_clue_breakdowns() -> list[dict]:
    """Load full secret definitions including clue_breakdown lists."""
    if not CLUE_BREAKDOWNS_PATH.exists():
        raise FileNotFoundError(f"{CLUE_BREAKDOWNS_PATH} not found.")
    secrets = json.loads(CLUE_BREAKDOWNS_PATH.read_text())
    for s in secrets:
        if "clue_breakdown" not in s or len(s["clue_breakdown"]) == 0:
            raise ValueError(f"Secret '{s['id']}' is missing 'clue_breakdown' field.")
    return secrets


def load_noise_topics() -> list[str]:
    return json.loads(NOISE_PATH.read_text()) if NOISE_PATH.exists() else DEFAULT_NOISE


def sample_secret():
    return random.choice(load_clue_breakdowns())


def sample_noise_topics(n: int):
    return random.choices(load_noise_topics(), k=n)