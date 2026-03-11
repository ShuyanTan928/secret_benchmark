"""
Registry of recommended sub-100B open models.
Run scripts/check_models.py to refresh from HuggingFace Hub.
"""

RECOMMENDED_MODELS = [
    {"name": "Qwen/Qwen1.5-32B-Chat",                    "params_b": 32, "context_len": 32768,  "notes": "Strong instruction following, multilingual"},
    {"name": "Qwen/Qwen1.5-72B-Chat",                    "params_b": 72, "context_len": 32768,  "notes": "Best Qwen quality under 100B"},
    {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1",     "params_b": 47, "context_len": 32768,  "notes": "Fast MoE architecture"},
    {"name": "meta-llama/Meta-Llama-3-70B-Instruct",     "params_b": 70, "context_len": 8192,   "notes": "Strong reasoning, Apache 2.0"},
    {"name": "microsoft/Phi-3-medium-128k-instruct",     "params_b": 14, "context_len": 128000, "notes": "Very long context, efficient"},
    {"name": "google/gemma-2-27b-it",                    "params_b": 27, "context_len": 8192,   "notes": "Google Gemma 2, competitive quality"},
]

def list_models(max_params_b: int = 100):
    return [m for m in RECOMMENDED_MODELS if m["params_b"] <= max_params_b]
