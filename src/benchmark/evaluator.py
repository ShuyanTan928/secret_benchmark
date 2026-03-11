import json
from pathlib import Path
from src.models.vllm_engine import VLLMEngine
from src.data_utils.schema import BenchmarkSample
from src.generation.prompts import EVALUATION_PROMPT


def format_thread(sample: BenchmarkSample) -> str:
    lines = []
    for dlg in sample.dialogues:
        for email in dlg.emails:
            lines.append(f"From: {email.sender}\nTo: {email.recipient}\nSubject: {email.subject}\n{email.body}\n")
            lines.append("=" * 40)
    return "\n".join(lines)


def evaluate_dataset(engine: VLLMEngine, dataset_path: str, output_path: str):
    samples = [BenchmarkSample(**json.loads(l)) for l in open(dataset_path)]
    prompts = [
        EVALUATION_PROMPT.format(person_a=s.person_a, person_b=s.person_b, email_thread=format_thread(s))
        for s in samples
    ]
    responses = engine.generate(prompts, max_tokens=256, temperature=0.0)
    results = []
    for s, resp in zip(samples, responses):
        found = "yes" in resp.lower().split("secret found:")[-1][:10]
        results.append({
            "sample_id": s.sample_id, "secret_topic": s.secret_topic,
            "snr": s.snr, "n_clues": s.n_clues, "n_noise": s.n_noise,
            "model_response": resp, "secret_found": found,
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {output_path}")
    return results
