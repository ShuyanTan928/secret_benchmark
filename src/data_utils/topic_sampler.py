import json, random
from pathlib import Path

SECRETS_PATH = Path("data/secrets.json")
NOISE_PATH = Path("data/noise_topics.json")

DEFAULT_NOISE = [
    "movies", "sports", "cooking", "travel", "music",
    "weekend plans", "pets", "fitness", "tech gadgets", "books",
]


def load_secrets():
    """Load secrets from JSON. Each secret must have 'clue_breakdown' field."""
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"{SECRETS_PATH} not found. Please create it with clue_breakdown fields."
        )
    secrets = json.loads(SECRETS_PATH.read_text())
    for s in secrets:
        if "clue_breakdown" not in s or len(s["clue_breakdown"]) == 0:
            raise ValueError(f"Secret '{s['id']}' is missing 'clue_breakdown' field.")
    return secrets


def load_noise_topics():
    return json.loads(NOISE_PATH.read_text()) if NOISE_PATH.exists() else DEFAULT_NOISE


def sample_secret():
    return random.choice(load_secrets())


def sample_noise_topics(n: int):
    return random.choices(load_noise_topics(), k=n)