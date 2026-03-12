"""
API-based inference engine for judge models.
Supports OpenAI-compatible APIs (OpenAI, Anthropic, Together, etc.)
"""
import os
from openai import OpenAI
from typing import List, Union


class APIEngine:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str = None,
        base_url: str = None,
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(response.choices[0].message.content.strip())
        return results