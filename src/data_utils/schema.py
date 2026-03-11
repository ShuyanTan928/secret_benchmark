from pydantic import BaseModel, Field
from typing import List, Optional


class EmailTurn(BaseModel):
    sender: str
    recipient: str
    subject: str
    body: str


class EmailDialogue(BaseModel):
    dialogue_id: str
    topic: str
    is_secret_clue: bool
    clue_index: Optional[int] = None
    emails: List[EmailTurn]


class BenchmarkSample(BaseModel):
    sample_id: str
    person_a: str
    person_b: str
    secret_topic: str
    secret_answer: str
    n_clues: int
    n_noise: int
    snr: float = Field(description="n_clues / (n_clues + n_noise)")
    dialogues: List[EmailDialogue]
    clue_positions: List[int]
