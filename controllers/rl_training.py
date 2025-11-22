import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.env_wrapper import make_env
from controllers.path_logger import PathLogger


class MetricLogger(BaseCallback):
    def __init__(self, csv_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists():
            self.csv_path.unlink()
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "episode_reward", "collision", "lane_offset"]
            )
            writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode"):
                data = {
                    "step": self.num_timesteps,
                    "episode_reward": info["episode"]["r"],
                    "collision": int(info.get("collision", False)),
                    "lane_offset": info.get("lane_offset", 0.0),
                }
                with self.csv_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    writer.writerow(data)
        return True


def train_local_agent(
    world_path: str,
    timesteps: int,
    log_dir: Path,
    learning_rate: float = 3e-4,
    log_paths: bool = False,
    path_log_interval: int = 10,
) -> Tuple[PPO, Path]:
    os.environ.setdefault("WEBOTS_CONTROLLER_URL", "ipc://")
    controller_url = os.environ["WEBOTS_CONTROLLER_URL"]
    if controller_url.startswith("ipc://") and controller_url.count(":") < 2:
        os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://127.0.0.1:6000/"  # default IPC endpoint
    env = DummyVecEnv([lambda: Monitor(make_env(display_progress=False), str(log_dir / "gym"))])
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        tensorboard_log=str(log_dir / "tb"),
        policy_kwargs=policy_kwargs,
    )
    logger = MetricLogger(log_dir / "metrics.csv")
    callbacks = [logger]
    
    # Optionally add path logger
    if log_paths:
        path_logger = PathLogger(log_dir / "paths", log_every_n_episodes=path_log_interval)
        callbacks.append(path_logger)
    
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]
    model.learn(total_timesteps=timesteps, callback=callback)
    checkpoint = log_dir / "ppo_webots.zip"
    model.save(checkpoint)
    env.close()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local PPO agent in Webots.")
    parser.add_argument("--world", type=str, default="webots_worlds/minimal_lane.wbt")
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/local"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-paths", action="store_true", help="Log car paths for visualization")
    parser.add_argument("--path-log-interval", type=int, default=10, help="Log path every N episodes")
    return parser.parse_args()


def main():
    torch.set_num_threads(max(torch.get_num_threads() // 2, 1))
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = train_local_agent(
        world_path=args.world,
        timesteps=args.timesteps,
        log_dir=args.log_dir,
        learning_rate=args.learning_rate,
        log_paths=args.log_paths,
        path_log_interval=args.path_log_interval,
    )
    print(f"Training complete. Checkpoint saved to {checkpoint}")


if __name__ == "__main__":
    main()

