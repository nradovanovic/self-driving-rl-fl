import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RL and FL metrics.")
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("metrics/figures"))
    parser.add_argument("--title", type=str, default="Training Metrics")
    parser.add_argument("--window", type=int, default=10, help="Moving average window size")
    return parser.parse_args()


def moving_average(data: pd.Series, window: int) -> pd.Series:
    """Compute moving average with proper handling of small datasets."""
    if len(data) < window:
        window = max(1, len(data) // 2)
    return data.rolling(window=window, min_periods=1).mean()


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

    # 1. Reward plot with moving average
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, df[reward_col], alpha=0.3, color="blue", label="Raw data (per episode/round)", linewidth=0.8)
    reward_ma = moving_average(df[reward_col], args.window)
    plt.plot(x_data, reward_ma, color="blue", label=f"Moving average (window={args.window})", linewidth=2)
    plt.xlabel(x_label + x_note)
    plt.ylabel("Reward")
    plt.title(f"{args.title} - Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    reward_path = args.output_dir / "reward.png"
    plt.savefig(reward_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Lane deviation plot
    if {"lane_offset", "lane_deviation"}.intersection(df.columns):
        deviation_col = "lane_offset" if "lane_offset" in df else "lane_deviation"
        plt.figure(figsize=(10, 5))
        abs_deviation = df[deviation_col].abs()
        plt.plot(x_data, abs_deviation, alpha=0.3, color="orange", label="Raw data (per episode/round)", linewidth=0.8)
        deviation_ma = moving_average(abs_deviation, args.window)
        plt.plot(x_data, deviation_ma, color="orange", label=f"Moving average (window={args.window})", linewidth=2)
        plt.xlabel(x_label + x_note)
        plt.ylabel("Deviation (abs)")
        plt.title(f"{args.title} - Lane Keeping")
        plt.grid(True, alpha=0.3)
        plt.legend()
        deviation_path = args.output_dir / "lane_deviation.png"
        plt.savefig(deviation_path, dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Collision rate plot
    if "collision_rate" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(x_data, df["collision_rate"], alpha=0.3, color="red", label="Raw data (per round)", linewidth=0.8)
        collision_ma = moving_average(df["collision_rate"], args.window)
        plt.plot(x_data, collision_ma, color="red", label=f"Moving average (window={args.window})", linewidth=2)
        plt.xlabel(x_label + x_note)
        plt.ylabel("Collision Rate")
        plt.title(f"{args.title} - Safety (Collision Rate)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, max(1.0, df["collision_rate"].max() * 1.1))
        collision_path = args.output_dir / "collision_rate.png"
        plt.savefig(collision_path, dpi=150, bbox_inches="tight")
        plt.close()
    elif "collision" in df.columns:
        # For local training, compute collision rate from binary collision column
        # Use a rolling window to compute rate
        window_size = min(args.window * 2, len(df) // 4) if len(df) > 10 else len(df)
        collision_rate = df["collision"].rolling(window=window_size, min_periods=1).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(x_data, collision_rate, color="red", label=f"Collision rate (rolling {window_size})", linewidth=2)
        plt.xlabel(x_label + x_note)
        plt.ylabel("Collision Rate")
        plt.title(f"{args.title} - Safety (Collision Rate)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, max(1.0, collision_rate.max() * 1.1))
        collision_path = args.output_dir / "collision_rate.png"
        plt.savefig(collision_path, dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Combined dashboard plot
    num_plots = 2  # reward + lane deviation (always available)
    if "collision_rate" in df.columns or "collision" in df.columns:
        num_plots += 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Reward subplot
    ax = axes[plot_idx]
    ax.plot(x_data, df[reward_col], alpha=0.3, color="blue", label="Raw data", linewidth=0.8)
    reward_ma = moving_average(df[reward_col], args.window)
    ax.plot(x_data, reward_ma, color="blue", label=f"MA-{args.window}", linewidth=2)
    ax.set_xlabel(x_label + x_note)
    ax.set_ylabel("Reward")
    ax.set_title("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_idx += 1
    
    # Lane deviation subplot
    if {"lane_offset", "lane_deviation"}.intersection(df.columns):
        deviation_col = "lane_offset" if "lane_offset" in df else "lane_deviation"
        ax = axes[plot_idx]
        abs_deviation = df[deviation_col].abs()
        ax.plot(x_data, abs_deviation, alpha=0.3, color="orange", label="Raw data", linewidth=0.8)
        deviation_ma = moving_average(abs_deviation, args.window)
        ax.plot(x_data, deviation_ma, color="orange", label=f"MA-{args.window}", linewidth=2)
        ax.set_xlabel(x_label + x_note)
        ax.set_ylabel("Lane Deviation (abs)")
        ax.set_title("Lane Keeping")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1
    
    # Collision rate subplot
    if "collision_rate" in df.columns:
        ax = axes[plot_idx]
        ax.plot(x_data, df["collision_rate"], alpha=0.3, color="red", label="Raw", linewidth=0.8)
        collision_ma = moving_average(df["collision_rate"], args.window)
        ax.plot(x_data, collision_ma, color="red", label=f"MA-{args.window}", linewidth=2)
        ax.set_xlabel(x_label + x_note)
        ax.set_ylabel("Collision Rate")
        ax.set_title("Safety (Collision Rate)")
        ax.set_ylim(0, max(1.0, df["collision_rate"].max() * 1.1))
        ax.grid(True, alpha=0.3)
        ax.legend()
    elif "collision" in df.columns:
        ax = axes[plot_idx]
        window_size = min(args.window * 2, len(df) // 4) if len(df) > 10 else len(df)
        collision_rate = df["collision"].rolling(window=window_size, min_periods=1).mean()
        ax.plot(x_data, collision_rate, color="red", label=f"Rolling {window_size}", linewidth=2)
        ax.set_xlabel(x_label + x_note)
        ax.set_ylabel("Collision Rate")
        ax.set_title("Safety (Collision Rate)")
        ax.set_ylim(0, max(1.0, collision_rate.max() * 1.1))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(args.title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    dashboard_path = args.output_dir / "dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()

