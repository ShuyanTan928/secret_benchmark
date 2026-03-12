"""
vLLM inference engine wrapper.
Unsloth note: Unsloth is for training (QLoRA/LoRA) only — not inference.
Workflow if fine-tuning: Unsloth → export HF format → serve with vLLM.
"""
from vllm import LLM, SamplingParams
from typing import List, Union


class VLLMEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]
