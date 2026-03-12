import json, random
from pathlib import Path

NAMES_PATH = Path("data/names.json")


def load_names() -> dict:
    if not NAMES_PATH.exists():
        raise FileNotFoundError(f"{NAMES_PATH} not found.")
    return json.loads(NAMES_PATH.read_text())


def sample_pair() -> tuple[str, str]:
    """Return (male_name, female_name). Alex -> male, Brooke -> female."""
    names = load_names()
    male = random.choice(names["male"])
    female = random.choice(names["female"])
    return male, female
