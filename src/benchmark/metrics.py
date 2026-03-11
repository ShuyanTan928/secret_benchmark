import json
import pandas as pd


def compute_metrics(results_path: str):
    df = pd.DataFrame(json.load(open(results_path)))
    print(f"\n=== Benchmark Metrics ===")
    print(f"Overall detection rate: {df['secret_found'].mean():.2%}")
    print("\nBy SNR:")
    print(df.groupby(pd.cut(df["snr"], bins=[0, 0.15, 0.3, 0.5, 1.0]))["secret_found"].mean().to_string())
    print("\nBy secret topic:")
    print(df.groupby("secret_topic")["secret_found"].mean().sort_values().to_string())
    return df
