import random, uuid
from src.data_utils.schema import BenchmarkSample, EmailDialogue, EmailTurn
from src.data_utils.person_sampler import sample_pair
from src.prompts import PLACEHOLDER_A, PLACEHOLDER_B


def _replace_names(dialogue: EmailDialogue, person_a: str, person_b: str) -> EmailDialogue:
    """Deep copy a dialogue and replace placeholder names with real names."""
    new_emails = []
    for email in dialogue.emails:
        new_emails.append(EmailTurn(
            sender=email.sender.replace(PLACEHOLDER_A, person_a).replace(PLACEHOLDER_B, person_b),
            recipient=email.recipient.replace(PLACEHOLDER_A, person_a).replace(PLACEHOLDER_B, person_b),
            subject=email.subject.replace(PLACEHOLDER_A, person_a).replace(PLACEHOLDER_B, person_b),
            body=email.body.replace(PLACEHOLDER_A, person_a).replace(PLACEHOLDER_B, person_b),
        ))
    return EmailDialogue(
        dialogue_id=dialogue.dialogue_id,
        topic=dialogue.topic,
        is_secret_clue=dialogue.is_secret_clue,
        clue_index=dialogue.clue_index,
        clue_description=dialogue.clue_description,
        emails=new_emails,
    )


def _insert_clues_non_adjacent(clue_dialogues: list, noise_dialogues: list) -> list:
    """Insert clue dialogues into noise list ensuring no two clues are adjacent."""
    combined = noise_dialogues.copy()
    n_clues = len(clue_dialogues)

    if n_clues == 0:
        return combined

    # We need at least (n_clues - 1) noise dialogues between clues
    if len(combined) < n_clues - 1:
        raise ValueError(
            f"Not enough noise ({len(combined)}) to separate {n_clues} clues. "
            f"Need at least {n_clues - 1} noise dialogues."
        )

    # Generate valid non-adjacent positions
    # Strategy: place clues one by one, ensuring gap >= 2 from previous
    for _ in range(100):  # max attempts
        positions = sorted(random.sample(range(len(combined) + n_clues), n_clues))
        # Check no two positions are adjacent
        valid = all(positions[i+1] - positions[i] >= 2 for i in range(len(positions) - 1))
        if valid:
            break
    else:
        # Fallback: evenly space them
        step = (len(combined) + n_clues) // (n_clues + 1)
        positions = [step * (i + 1) for i in range(n_clues)]

    for pos, dlg in zip(positions, clue_dialogues):
        combined.insert(pos, dlg)

    return combined


def combine(
    secret_id: str,
    secret_label: str,
    clue_dialogues: list[EmailDialogue],
    noise_pool: list[EmailDialogue],
    n_noise: int,
) -> BenchmarkSample:
    """Combine clue and noise dialogues with random gendered names."""
    person_a, person_b = sample_pair()  

    # Sample noise WITHOUT replacement
    n_noise = min(n_noise, len(noise_pool))
    noise_sample = random.sample(noise_pool, n_noise)

    # Replace placeholders with real names
    named_clues = [_replace_names(d, person_a, person_b) for d in clue_dialogues]
    named_noise = [_replace_names(d, person_a, person_b) for d in noise_sample]

    # Assign clue indices
    for idx, dlg in enumerate(named_clues):
        dlg.clue_index = idx + 1

    # Insert clues at non-adjacent positions
    combined = _insert_clues_non_adjacent(named_clues, named_noise)

    n_clues = len(named_clues)
    return BenchmarkSample(
        sample_id=str(uuid.uuid4())[:12],
        person_a=person_a,
        person_b=person_b,
        secret_topic=secret_id,
        secret_answer=secret_label,
        n_clues=n_clues,
        n_noise=n_noise,
        snr=round(n_clues / (n_clues + n_noise), 3),
        dialogues=combined,
        clue_positions=[i for i, d in enumerate(combined) if d.is_secret_clue],
    )