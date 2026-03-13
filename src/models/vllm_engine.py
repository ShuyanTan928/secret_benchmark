"""
vLLM inference engine wrapper for the email benchmark.

Key design decisions:
  - Uses chat template with enable_thinking=False for clean outputs
  - Applies chat template at generation time so prompts stay as plain strings
    throughout the pipeline (no special tokens in eval_prompts.py)
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class VLLMEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        enable_thinking: bool = False,
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking

        # Load tokenizer separately for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def _apply_chat_template(self, prompt: str) -> str:
        """
        Wrap a plain-text prompt into the model's chat template.

        This converts:
            "Is there a secret? Answer Yes or No."
        Into the full chat-formatted string with special tokens, e.g.:
            <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n

        Setting enable_thinking=False tells Qwen3 to skip the <think> block
        and output the answer directly.
        """
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return formatted

    def generate(
        self,
        prompt: str | list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: list[str] | None = None,
    ) -> list[str]:
        """
        Generate completions for one or more prompts.

        Args:
            prompt: single prompt string or list of prompt strings
            max_tokens: maximum tokens to generate
            temperature: sampling temperature (0.0 = greedy)
            top_p: nucleus sampling parameter
            top_k: top-k sampling (-1 = disabled)
            stop: optional stop sequences

        Returns:
            list of generated text strings (one per prompt)
        """
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Apply chat template to each prompt
        formatted_prompts = [self._apply_chat_template(p) for p in prompts]

        # For non-thinking mode with greedy decoding, keep temp at 0
        # For thinking mode, Qwen3 recommends temp=0.6, top_p=0.95, top_k=20
        if self.enable_thinking and temperature == 0.0:
            temperature = 0.6
            top_p = 0.95
            top_k = 20

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )

        outputs = self.llm.generate(
            formatted_prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        results = []
        for output in outputs:
            text = output.outputs[0].text
            results.append(text)

        return results

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> list[str]:
        """Convenience method: batch generate with default params."""
        return self.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )