import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RL and FL metrics.")
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("metrics/figures"))
    parser.add_argument("--title", type=str, default="Training Metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_file)

    if "episode_reward" in df:
        reward_col = "episode_reward"
    else:
        reward_col = "reward"
    
    # Detect if this is federated learning (has "round" column) or local training
    is_federated = "round" in df.columns
    
    # Determine x-axis data and label
    if is_federated:
        # Federated learning: use round number
        x_data = df["round"]
        x_label = "Federated Learning Round"
        x_note = " (evaluation after each round)"
    elif "step" in df:
        # Local training: use step number
        x_data = df["step"]
        x_label = "Training Step"
        x_note = " (per episode during training)"
    else:
        # Fallback: use index
        x_data = df.index
        x_label = "Episode / Round"
        x_note = ""

    plt.figure(figsize=(8, 4))
    plt.plot(x_data, df[reward_col], label="Reward")
    plt.xlabel(x_label + x_note)
    plt.ylabel("Reward")
    plt.title(f"{args.title} - Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    reward_path = args.output_dir / "reward.png"
    plt.savefig(reward_path, dpi=150, bbox_inches="tight")

    if {"lane_offset", "lane_deviation"}.intersection(df.columns):
        deviation_col = "lane_offset" if "lane_offset" in df else "lane_deviation"
        plt.figure(figsize=(8, 4))
        plt.plot(x_data, df[deviation_col].abs(), color="orange", label="Lane deviation")
        plt.xlabel(x_label + x_note)
        plt.ylabel("Deviation (abs)")
        plt.title(f"{args.title} - Lane Keeping")
        plt.grid(True, alpha=0.3)
        plt.legend()
        deviation_path = args.output_dir / "lane_deviation.png"
        plt.savefig(deviation_path, dpi=150, bbox_inches="tight")

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()

