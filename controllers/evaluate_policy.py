import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.env_wrapper import make_env  # noqa: E402


def run_episode(model: PPO, deterministic: bool = True) -> None:
    env = make_env(auto_connect=True, display_progress=True)
    obs, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0.0
    step = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step += 1
        if step % 50 == 0:
            print(
                f"[step {step:04d}] reward={reward: .3f} "
                f"speed={info.get('speed', np.nan):.3f} "
                f"offset={info.get('lane_offset', np.nan):.3f}"
            )
    env.close()
    print(f"Episode finished after {step} steps. Total reward: {episode_reward:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/local_client/ppo_webots.zip"),
        help="Path to the saved PPO checkpoint.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions during evaluation (default deterministic).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    print(f"Loading model from {args.checkpoint}")
    model = PPO.load(args.checkpoint)
    run_episode(model, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()

