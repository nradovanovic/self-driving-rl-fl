"""
Visualize car paths on the elliptic road to show learning progress.
Shows paths from early (bad) episodes to late (good) episodes.
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize car paths on elliptic road showing learning progress."
    )
    parser.add_argument(
        "--paths-dir",
        type=Path,
        required=True,
        help="Directory containing episode path JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/figures/path_progression.png"),
        help="Output image path"
    )
    parser.add_argument(
        "--ellipse-a",
        type=float,
        default=5.25,
        help="Ellipse semi-major axis (a)"
    )
    parser.add_argument(
        "--ellipse-b",
        type=float,
        default=2.25,
        help="Ellipse semi-minor axis (b)"
    )
    parser.add_argument(
        "--width-ratio",
        type=float,
        default=0.14285714285714285,
        help="Lane width ratio"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to show (None = all)"
    )
    parser.add_argument(
        "--episode-stride",
        type=int,
        default=1,
        help="Show every Nth episode (1 = all, 10 = every 10th)"
    )
    return parser.parse_args()


def draw_ellipse_road(
    ax,
    a: float,
    b: float,
    width_ratio: float,
    num_points: int = 200
) -> None:
    """Draw the elliptic road (centerline and lane boundaries)."""
    theta = np.linspace(0, 2 * math.pi, num_points)
    
    # Centerline
    centerline_x = a * np.cos(theta)
    centerline_z = b * np.sin(theta)
    ax.plot(centerline_x, centerline_z, 'k--', linewidth=1, alpha=0.5, label='Centerline')
    
    # Inner boundary (left side)
    inner_x = centerline_x - width_ratio * a * np.cos(theta)
    inner_z = centerline_z - width_ratio * a * np.sin(theta)
    ax.plot(inner_x, inner_z, 'w-', linewidth=2, alpha=0.8)
    
    # Outer boundary (right side)
    outer_x = centerline_x + width_ratio * a * np.cos(theta)
    outer_z = centerline_z + width_ratio * a * np.sin(theta)
    ax.plot(outer_x, outer_z, 'w-', linewidth=2, alpha=0.8)
    
    # Fill road area
    road_x = np.concatenate([inner_x, outer_x[::-1]])
    road_z = np.concatenate([inner_z, outer_z[::-1]])
    ax.fill(road_x, road_z, color='gray', alpha=0.3, zorder=0)


def load_paths(paths_dir: Path, max_episodes: int = None, stride: int = 1) -> List[dict]:
    """Load episode paths from JSON files."""
    path_files = sorted(paths_dir.glob("episode_*.json"))
    
    if max_episodes:
        path_files = path_files[:max_episodes]
    
    paths = []
    for path_file in path_files[::stride]:
        try:
            with path_file.open() as f:
                data = json.load(f)
                paths.append(data)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    
    return paths


def visualize_paths(
    paths: List[dict],
    output_path: Path,
    ellipse_a: float,
    ellipse_b: float,
    width_ratio: float
) -> None:
    """Create visualization showing path progression."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Draw the road
    draw_ellipse_road(ax, ellipse_a, ellipse_b, width_ratio)
    
    # Color scheme: early episodes (bad) = red, late episodes (good) = green
    num_paths = len(paths)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, num_paths)) if num_paths > 1 else ['blue']
    
    # Plot each path
    for idx, path_data in enumerate(paths):
        episode = path_data.get("episode", idx)
        path = path_data.get("path", [])
        collision = path_data.get("collision", False)
        reward = path_data.get("reward", 0.0)
        
        if len(path) < 2:
            continue
        
        # Extract x, z coordinates
        x_coords = [p[0] for p in path]
        z_coords = [p[1] for p in path]
        
        # Determine line style based on collision
        linestyle = '-' if not collision else '--'
        linewidth = 1.5 if not collision else 1.0
        alpha = 0.7 if not collision else 0.4
        
        # Plot path
        color = colors[idx] if num_paths > 1 else colors[0]
        ax.plot(
            x_coords, z_coords,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1
        )
        
        # Mark start point
        if len(x_coords) > 0:
            ax.scatter(
                x_coords[0], z_coords[0],
                color=color,
                marker='o',
                s=30,
                zorder=2,
                edgecolors='black',
                linewidths=0.5
            )
    
    # Add colorbar legend if multiple paths
    if num_paths > 1:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=num_paths-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Episode Number (Early → Late)')
        cbar.set_ticks([0, num_paths-1])
        cbar.set_ticklabels(['Early (Bad)', 'Late (Good)'])
    
    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Z Position (m)', fontsize=12)
    ax.set_title(
        f'Car Path Progression on Elliptic Road\n'
        f'Showing {num_paths} episodes (Red=Early/Bad, Green=Late/Good)',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add episode info text
    if paths:
        first_ep = paths[0].get("episode", 0)
        last_ep = paths[-1].get("episode", 0)
        info_text = f"Episodes: {first_ep} → {last_ep}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved path visualization to {output_path}")
    print(f"  - Total episodes shown: {num_paths}")
    if paths:
        print(f"  - Episode range: {paths[0].get('episode', 0)} to {paths[-1].get('episode', 0)}")


def main():
    args = parse_args()
    
    if not args.paths_dir.exists():
        raise FileNotFoundError(f"Paths directory not found: {args.paths_dir}")
    
    # Load paths
    paths = load_paths(args.paths_dir, args.max_episodes, args.episode_stride)
    
    if not paths:
        print(f"No path files found in {args.paths_dir}")
        return
    
    # Create visualization
    visualize_paths(
        paths,
        args.output,
        args.ellipse_a,
        args.ellipse_b,
        args.width_ratio
    )


if __name__ == "__main__":
    main()

