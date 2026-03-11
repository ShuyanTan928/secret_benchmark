import random, uuid
from src.data_utils.schema import BenchmarkSample, EmailDialogue


def combine(
    person_a: str, person_b: str,
    secret_topic: str, secret_answer: str,
    clue_dialogues: list[EmailDialogue],
    noise_dialogues: list[EmailDialogue],
) -> BenchmarkSample:
    n_clues, n_noise = len(clue_dialogues), len(noise_dialogues)
    combined = noise_dialogues.copy()
    positions = sorted(random.sample(range(len(combined) + n_clues), n_clues))
    for pos, dlg in zip(positions, clue_dialogues):
        combined.insert(pos, dlg)
    return BenchmarkSample(
        sample_id=str(uuid.uuid4())[:12],
        person_a=person_a, person_b=person_b,
        secret_topic=secret_topic, secret_answer=secret_answer,
        n_clues=n_clues, n_noise=n_noise,
        snr=round(n_clues / (n_clues + n_noise), 3),
        dialogues=combined,
        clue_positions=[i for i, d in enumerate(combined) if d.is_secret_clue],
    )
