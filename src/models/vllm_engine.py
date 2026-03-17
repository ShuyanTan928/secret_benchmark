"""
vLLM inference engine wrapper for the email benchmark.

Key design decisions:
  - Uses chat template with enable_thinking=False for clean outputs
  - Applies chat template at generation time so prompts stay as plain strings
    throughout the pipeline (no special tokens in eval_prompts.py)
  - Supports quantization (4bit/8bit) via GPTQ/AWQ/FP8 for larger models
  - Pre-defined model configs with model-specific system prompts and sampling
"""

import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Phi-4-reasoning system prompt (required by the model)
# ---------------------------------------------------------------------------
PHI4_REASONING_SYSTEM_PROMPT = (
    "You are Phi, a language model trained by Microsoft to help users. "
    "Your role as an assistant involves thoroughly exploring questions through "
    "a systematic thinking process before providing the final precise and "
    "accurate solutions. This requires engaging in a comprehensive cycle of "
    "analysis, summarizing, exploration, reassessment, reflection, backtracing, "
    "and iteration to develop well-considered thinking process. Please structure "
    "your response into two main sections: Thought and Solution using the "
    "specified format: <think> {Thought section} </think> {Solution section}. "
    "In the Thought section, detail your reasoning process in steps. Each step "
    "should include detailed considerations such as analysing questions, "
    "summarizing relevant findings, brainstorming new ideas, verifying the "
    "accuracy of the current steps, refining any errors, and revisiting "
    "previous steps. In the Solution section, based on various attempts, "
    "explorations, and reflections from the Thought section, systematically "
    "present the final solution that you deem correct. The Solution section "
    "should be logical, accurate, and concise and detail necessary steps "
    "needed to reach the conclusion. Now, try to solve the following question "
    "through the above guidelines:"
)


# ---------------------------------------------------------------------------
# Pre-defined model configs
#
# Each config can specify:
#   model_name          - HuggingFace model ID
#   tensor_parallel_size - number of GPUs
#   max_model_len       - context window
#   quantization        - None, "gptq_marlin", "fp8", etc.
#   is_reasoning        - True for models that output <think>...</think>
#   system_prompt       - model-specific system prompt (None = no system msg)
#   sampling            - model-specific default sampling overrides
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    # ---- Qwen3 family ----
    "qwen3-14b": {
        "model_name": "Qwen/Qwen3-14B",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "quantization": None,
    },
    "qwen3-32b": {
        "model_name": "Qwen/Qwen3-32B",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "quantization": None,
    },
    "qwen3-235b-4bit": {
        "model_name": "Qwen/Qwen3-235B-A22B-GPTQ-Int4",
        "tensor_parallel_size": 2,
        "max_model_len": 32768,
        "quantization": "gptq_marlin",
    },
    "qwen3-235b-8bit": {
        "model_name": "QuantTrio/Qwen3-235B-A22B-GPTQ-Int8",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "quantization": "gptq_marlin",
    },
    "qwen3-235b-fp8": {
        "model_name": "Qwen/Qwen3-235B-A22B-FP8",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "quantization": "fp8",
    },

    # ---- Llama 4 Maverick (400B MoE, 17B active) ----
    "llama4-maverick-fp8": {
        "model_name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "quantization": "fp8",
    },
    "llama4-maverick": {
        "model_name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "quantization": None,
    },

    # ---- GPT-oss-120B (117B dense) ----
    "gpt-oss-120b": {
        "model_name": "openai/GPT-oss-120B",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "quantization": None,
    },

    # ---- Gemma 3 27B (27B dense) ----
    "gemma3-27b": {
        "model_name": "google/gemma-3-27b-it",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "quantization": None,
    },

    # ---- Phi-4 reasoning plus (14B dense, reasoning model) ----
    "phi4-reasoning-plus": {
        "model_name": "microsoft/Phi-4-reasoning-plus",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "quantization": None,
        "is_reasoning": True,
        "system_prompt": PHI4_REASONING_SYSTEM_PROMPT,
        "sampling": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
        },
    },
}


def list_presets() -> str:
    """Return a formatted string of all available model presets."""
    lines = []
    for key, cfg in MODEL_CONFIGS.items():
        q = cfg.get("quantization") or "none"
        r = " [reasoning]" if cfg.get("is_reasoning") else ""
        lines.append(
            f"  {key:<25s} tp={cfg['tensor_parallel_size']}  "
            f"quant={q:<15s} {cfg['model_name']}{r}"
        )
    return "\n".join(lines)


class VLLMEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        enable_thinking: bool = False,
        quantization: str | None = None,
        is_reasoning: bool = False,
        system_prompt: str | None = None,
        default_sampling: dict | None = None,
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.is_reasoning = is_reasoning
        self.system_prompt = system_prompt
        self.default_sampling = default_sampling or {}

        # Load tokenizer separately for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Build vLLM kwargs
        llm_kwargs = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if quantization:
            llm_kwargs["quantization"] = quantization

        # Initialize vLLM engine
        self.llm = LLM(**llm_kwargs)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        gpu_memory_utilization: float = 0.9,
        enable_thinking: bool = False,
        **overrides,
    ) -> "VLLMEngine":
        """Create engine from a pre-defined model config.

        Usage:
            engine = VLLMEngine.from_preset("qwen3-235b-4bit")
            engine = VLLMEngine.from_preset("phi4-reasoning-plus")

        Override any config value:
            engine = VLLMEngine.from_preset("qwen3-235b-4bit", max_model_len=16384)
        """
        if preset not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available presets:\n{list_presets()}"
            )

        config = {**MODEL_CONFIGS[preset], **overrides}
        return cls(
            model_name=config["model_name"],
            tensor_parallel_size=config["tensor_parallel_size"],
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=gpu_memory_utilization,
            enable_thinking=enable_thinking,
            quantization=config.get("quantization"),
            is_reasoning=config.get("is_reasoning", False),
            system_prompt=config.get("system_prompt"),
            default_sampling=config.get("sampling"),
        )

    def _apply_chat_template(self, prompt: str) -> str:
        """Format prompt with chat template, including system prompt if set."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Phi-4 reasoning doesn't use enable_thinking kwarg
        if self.is_reasoning:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        return formatted

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from reasoning model output.
        Handles both closed and unclosed think tags."""
        # Case 1: closed <think>...</think> — extract content after it
        match_closed = re.search(r"</think>\s*(.*)", text, flags=re.DOTALL)
        if match_closed and match_closed.group(1).strip():
            return match_closed.group(1).strip()

        # Case 2: no </think> found — model ran out of tokens while thinking
        # Remove everything from <think> onward
        stripped = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
        if stripped:
            return stripped

        # Case 3: entire output is inside <think> with no answer
        return ""

    def generate(
        self,
        prompt: str | list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: list[str] | None = None,
    ) -> list[str]:

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Apply chat template to each prompt
        formatted_prompts = [self._apply_chat_template(p) for p in prompts]

        # Reasoning models need more tokens for the thinking chain
        if self.is_reasoning and max_tokens < 16384:
            max_tokens = 16384

        # Apply model-specific sampling defaults, then caller overrides
        # Priority: model defaults < caller args (if non-default)
        final_temp = self.default_sampling.get("temperature", temperature)
        final_top_p = self.default_sampling.get("top_p", top_p)
        final_top_k = self.default_sampling.get("top_k", top_k)

        # If caller explicitly set temperature (non-zero), respect it
        if temperature != 0.0:
            final_temp = temperature

        # For Qwen thinking mode, use Qwen-recommended params
        if self.enable_thinking and not self.is_reasoning and temperature == 0.0:
            final_temp = 0.6
            final_top_p = 0.95
            final_top_k = 20

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=final_temp,
            top_p=final_top_p,
            top_k=final_top_k,
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
            # Strip thinking traces from reasoning models
            if self.is_reasoning:
                clean = self._strip_thinking(text)
                text = clean if clean else text
            results.append(text)

        return results

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> list[str]:

        return self.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )