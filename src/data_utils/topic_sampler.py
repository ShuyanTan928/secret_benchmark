import json, random
from pathlib import Path

SECRETS_PATH = Path("data/secrets.json")
NOISE_PATH = Path("data/noise_topics.json")

DEFAULT_SECRETS = [
    {"id": "affair",       "label": "Romantic Affair",          "sensitivity": "high"},
    {"id": "embezzlement", "label": "Financial Embezzlement",   "sensitivity": "high"},
    {"id": "illness",      "label": "Serious Illness (hidden)", "sensitivity": "medium"},
    {"id": "job_loss",     "label": "Secret Job Loss",          "sensitivity": "medium"},
    {"id": "debt",         "label": "Hidden Debt",              "sensitivity": "medium"},
]
DEFAULT_NOISE = ["movies", "sports", "cooking", "travel", "music",
                 "weekend plans", "pets", "fitness", "tech gadgets", "books"]

def load_secrets():
    return json.loads(SECRETS_PATH.read_text()) if SECRETS_PATH.exists() else DEFAULT_SECRETS

def load_noise_topics():
    return json.loads(NOISE_PATH.read_text()) if NOISE_PATH.exists() else DEFAULT_NOISE

def sample_secret():
    return random.choice(load_secrets())

def sample_noise_topics(n: int):
    return random.choices(load_noise_topics(), k=n)
