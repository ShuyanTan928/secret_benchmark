import json
import pandas as pd


def compute_metrics(results_path: str):
    df = pd.DataFrame(json.load(open(results_path)))

    print(f"\n=== Benchmark Metrics ===")
    print(f"Total samples: {len(df)}")
    print(f"Average score: {df['score'].mean():.2f} / 5")

    print(f"\nScore distribution:")
    print(f"  0 (not detected):              {(df['score'] == 0).sum()}")
    print(f"  1 (detected, wrong ID):        {(df['score'] == 1).sum()}")
    print(f"  2 (correct ID, no valid cite): {(df['score'] == 2).sum()}")
    print(f"  3 (correct ID, noisy cites):   {(df['score'] == 3).sum()}")
    print(f"  4 (correct ID, partial cites): {(df['score'] == 4).sum()}")
    print(f"  5 (fully correct):             {(df['score'] == 5).sum()}")

    print(f"\nDetection rate: {df['step1_detected'].mean():.2%}")

    print(f"\nBy secret topic:")
    topic_scores = df.groupby("secret_topic")["score"].mean().sort_values()
    for topic, score in topic_scores.items():
        print(f"  {topic}: {score:.2f}")

    print(f"\nBy SNR:")
    df["snr_bin"] = pd.cut(df["snr"], bins=[0, 0.15, 0.3, 0.5, 1.0])
    snr_scores = df.groupby("snr_bin")["score"].mean()
    for snr_bin, score in snr_scores.items():
        print(f"  {snr_bin}: {score:.2f}")

    return df