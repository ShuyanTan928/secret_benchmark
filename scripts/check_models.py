"""
Query HuggingFace Hub for latest text-generation models under 100B params.
Usage: python scripts/check_models.py --max_params 100 --top_k 20
"""
import argparse
from huggingface_hub import HfApi


def search_models(max_params_b: int = 100, top_k: int = 20):
    api = HfApi()
    models = api.list_models(task="text-generation", sort="downloads", direction=-1, limit=200, cardData=True)
    results = []
    for m in models:
        if any(x in m.modelId.lower() for x in ["405b", "340b", "180b", "141b", "110b"]):
            continue
        results.append({"id": m.modelId, "downloads": m.downloads, "likes": m.likes})
        if len(results) >= top_k:
            break
    print(f"\nTop {top_k} models on HuggingFace (<~{max_params_b}B):\n")
    for i, r in enumerate(results, 1):
        print(f"  {i:2}. {r['id']}  (↓{r['downloads']:,}  ♥{r['likes']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_params", type=int, default=100)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()
    search_models(args.max_params, args.top_k)
