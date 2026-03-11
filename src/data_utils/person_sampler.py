import json, random
from pathlib import Path

NAMES_PATH = Path("data/names.json")

DEFAULT_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "James", "Karen", "Leo", "Mia", "Nathan", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe", "Aaron", "Bella", "Chris", "Diana",
]

def load_names() -> list:
    if NAMES_PATH.exists():
        return json.loads(NAMES_PATH.read_text())
    return DEFAULT_NAMES

def sample_pair() -> tuple[str, str]:
    names = load_names()
    a, b = random.sample(names, 2)
    return a, b
