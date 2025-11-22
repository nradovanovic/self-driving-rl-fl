"""
Compare local training and federated learning metrics side by side.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local training and federated learning metrics."
    )
    parser.add_argument(
        "--local-metrics",
        type=Path,
        help="Path to local training metrics CSV (e.g., runs/local_client/metrics.csv)",
    )
    parser.add_argument(
        "--federated-metrics",
        type=Path,
        help="Path to federated learning metrics CSV (e.g., runs/federated/client01/metrics.csv)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("metrics/figures")
    )
    parser.add_argument("--title", type=str, default="Training Comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    local_df = None
    federated_df = None

    if args.local_metrics and args.local_metrics.exists():
        local_df = pd.read_csv(args.local_metrics)
        print(f"Loaded local training metrics: {len(local_df)} episodes")

    if args.federated_metrics and args.federated_metrics.exists():
        federated_df = pd.read_csv(args.federated_metrics)
        print(f"Loaded federated learning metrics: {len(federated_df)} rounds")

    if local_df is None and federated_df is None:
        raise ValueError("At least one metrics file must be provided")

    # Determine reward column names and normalize x-axis to progress (0-1)
    local_reward_col = None
    federated_reward_col = None
    local_progress = None
    federated_progress = None

    if local_df is not None:
        local_reward_col = (
            "episode_reward" if "episode_reward" in local_df else "reward"
        )
        # Normalize to progress (0-1) for fair comparison
        local_progress = local_df.index / max(len(local_df) - 1, 1)

    if federated_df is not None:
        federated_reward_col = (
            "episode_reward" if "episode_reward" in federated_df else "reward"
        )
        # Normalize to progress (0-1) for fair comparison
        federated_progress = federated_df.index / max(len(federated_df) - 1, 1)

    # Plot 1: Reward comparison (normalized to progress)
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if local_df is not None:
        ax.plot(
            local_progress * 100,  # Convert to percentage for readability
            local_df[local_reward_col],
            label="Local Training",
            color="blue",
            alpha=0.7,
            linewidth=1.5,
        )
    
    if federated_df is not None:
        ax.plot(
            federated_progress * 100,  # Convert to percentage for readability
            federated_df[federated_reward_col],
            label="Federated Learning",
            color="green",
            alpha=0.7,
            linewidth=1.5,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Training Progress (%)", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.set_title(f"{args.title} - Reward Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    reward_path = args.output_dir / "reward_comparison.png"
    plt.savefig(reward_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reward comparison to {reward_path}")

    # Plot 2: Lane deviation comparison
    local_deviation_col = None
    federated_deviation_col = None

    if local_df is not None:
        if "lane_offset" in local_df:
            local_deviation_col = "lane_offset"
        elif "lane_deviation" in local_df:
            local_deviation_col = "lane_deviation"

    if federated_df is not None:
        if "lane_offset" in federated_df:
            federated_deviation_col = "lane_offset"
        elif "lane_deviation" in federated_df:
            federated_deviation_col = "lane_deviation"

    if local_deviation_col or federated_deviation_col:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if local_df is not None and local_deviation_col:
            ax.plot(
                local_progress * 100,  # Convert to percentage
                local_df[local_deviation_col].abs(),
                label="Local Training",
                color="blue",
                alpha=0.7,
                linewidth=1.5,
            )
        
        if federated_df is not None and federated_deviation_col:
            ax.plot(
                federated_progress * 100,  # Convert to percentage
                federated_df[federated_deviation_col].abs(),
                label="Federated Learning",
                color="green",
                alpha=0.7,
                linewidth=1.5,
                marker="o",
                markersize=4,
            )

        ax.set_xlabel("Training Progress (%)", fontsize=11)
        ax.set_ylabel("Lane Deviation (abs)", fontsize=11)
        ax.set_title(
            f"{args.title} - Lane Keeping Comparison", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()
        deviation_path = args.output_dir / "lane_deviation_comparison.png"
        plt.savefig(deviation_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved lane deviation comparison to {deviation_path}")

    # Create side-by-side subplot version (normalized to progress)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward subplot
    ax = axes[0]
    if local_df is not None:
        ax.plot(
            local_progress * 100,  # Convert to percentage
            local_df[local_reward_col],
            label="Local Training",
            color="blue",
            alpha=0.7,
            linewidth=1.5,
        )
    if federated_df is not None:
        ax.plot(
            federated_progress * 100,  # Convert to percentage
            federated_df[federated_reward_col],
            label="Federated Learning",
            color="green",
            alpha=0.7,
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
    ax.set_xlabel("Training Progress (%)", fontsize=10)
    ax.set_ylabel("Reward", fontsize=10)
    ax.set_title("Reward", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Lane deviation subplot
    ax = axes[1]
    if local_df is not None and local_deviation_col:
        ax.plot(
            local_progress * 100,  # Convert to percentage
            local_df[local_deviation_col].abs(),
            label="Local Training",
            color="blue",
            alpha=0.7,
            linewidth=1.5,
        )
    if federated_df is not None and federated_deviation_col:
        ax.plot(
            federated_progress * 100,  # Convert to percentage
            federated_df[federated_deviation_col].abs(),
            label="Federated Learning",
            color="green",
            alpha=0.7,
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
    ax.set_xlabel("Training Progress (%)", fontsize=10)
    ax.set_ylabel("Lane Deviation (abs)", fontsize=10)
    ax.set_title("Lane Keeping", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.suptitle(f"{args.title}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    comparison_path = args.output_dir / "training_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side comparison to {comparison_path}")


if __name__ == "__main__":
    main()

