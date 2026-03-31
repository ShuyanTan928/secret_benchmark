"""
API-based inference engine for judge models and API-based testers.
Supports OpenAI-compatible APIs (OpenAI, Google Gemini/Gemma, etc.)
Includes automatic retry with backoff for rate limits.
"""
import os
import time
import re
from openai import OpenAI
from typing import List, Union
from dotenv import load_dotenv
load_dotenv()


class APIEngine:
    def __init__(
        self,
        model_name: str = "gemma-3-27b-it",
        api_key: str = None,
        base_url: str = None,
        max_retries: int = 5,
        retry_base_delay: float = 60.0,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def _call_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Single API call with retry on rate limit errors."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "RateLimitError" in type(e).__name__:
                    # Try to extract retry delay from error message
                    match = re.search(r"retry in (\d+\.?\d*)", error_str, re.IGNORECASE)
                    if match:
                        wait = float(match.group(1)) + 5
                    else:
                        wait = self.retry_base_delay * (attempt + 1)
                    print(f"    [Rate limit] Waiting {wait:.0f}s before retry {attempt+1}/{self.max_retries}...")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"API call failed after {self.max_retries} retries for model {self.model_name}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            result = self._call_api(prompt, max_tokens, temperature)
            results.append(result)
        return results