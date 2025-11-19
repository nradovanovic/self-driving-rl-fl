import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate training metrics.")
    parser.add_argument("--metrics-file", type=Path, required=True, help="CSV file to load.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.metrics_file.exists():
        raise FileNotFoundError(args.metrics_file)
    df = pd.read_csv(args.metrics_file)
    summary = {
        "episodes": len(df),
        "avg_reward": df["episode_reward"].mean() if "episode_reward" in df else df["reward"].mean(),
        "collision_rate": df.get("collision", pd.Series(dtype=float)).mean()
        if "collision" in df
        else df.get("collision_rate", pd.Series(dtype=float)).mean(),
        "avg_lane_offset": df.get("lane_offset", pd.Series(dtype=float)).abs().mean()
        if "lane_offset" in df
        else df.get("lane_deviation", pd.Series(dtype=float)).mean(),
    }
    print("=== Metric Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()

