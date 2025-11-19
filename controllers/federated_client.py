import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.env_wrapper import make_env


def model_to_list(model: PPO) -> List[np.ndarray]:
    return [param.detach().cpu().numpy() for param in model.policy.parameters()]


def list_to_model(model: PPO, parameters: List[np.ndarray]) -> None:
    with torch.no_grad():
        for p, new in zip(model.policy.parameters(), parameters):
            p.data = torch.from_numpy(new).to(p.device)


def evaluate_agent(model: PPO, episodes: int = 5) -> Dict[str, float]:
    env = DummyVecEnv([lambda: Monitor(make_env(display_progress=False), None)])
    rewards = []
    collisions = 0
    lane_offsets = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            ep_reward += float(reward[0])
            done = bool(dones[0])
            if infos[0].get("collision"):
                collisions += 1
            lane_offsets.append(abs(infos[0].get("lane_offset", 0.0)))
        rewards.append(ep_reward)
    env.close()
    return {
        "reward": float(np.mean(rewards)),
        "collision_rate": float(collisions / max(episodes, 1)),
        "lane_deviation": float(np.mean(lane_offsets)),
    }


class WebotsFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        world_path: str,
        log_dir: Path,
        round_timesteps: int,
        learning_rate: float,
    ) -> None:
        self.client_id = client_id
        self.world_path = world_path
        self.log_dir = log_dir
        self.round_timesteps = round_timesteps
        self.learning_rate = learning_rate
        self.log_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("WEBOTS_CONTROLLER_URL", "ipc://")
        controller_url = os.environ["WEBOTS_CONTROLLER_URL"]
        if controller_url.startswith("ipc://") and controller_url.count(":") < 2:
            os.environ["WEBOTS_CONTROLLER_URL"] = "ipc://127.0.0.1:6000/"

        self.env = DummyVecEnv([lambda: Monitor(make_env(display_progress=False), None)])
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_rate=self.learning_rate,
            policy_kwargs={"net_arch": [64, 64]},
            tensorboard_log=str(self.log_dir / "tb"),
        )
        self.metrics_file = self.log_dir / "metrics.csv"
        if not self.metrics_file.exists():
            with self.metrics_file.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "round",
                        "reward",
                        "collision_rate",
                        "lane_deviation",
                    ],
                )
                writer.writeheader()

    # Flower interface -------------------------------------------------------
    def get_parameters(self, config):  # type: ignore[override]
        return model_to_list(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        round_number = config.get("server_round", 0)
        list_to_model(self.model, parameters)
        self.model.learn(total_timesteps=self.round_timesteps)
        metrics = evaluate_agent(self.model, episodes=3)
        with self.metrics_file.open("a", newline="") as f:
            fieldnames = ["round", "reward", "collision_rate", "lane_deviation"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({"round": round_number, **metrics})
        return self.get_parameters(config), self.round_timesteps, metrics

    def evaluate(self, parameters, config):  # type: ignore[override]
        list_to_model(self.model, parameters)
        metrics = evaluate_agent(self.model, episodes=3)
        return float(metrics["reward"]), self.round_timesteps, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webots Federated Learning client.")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--client-id", type=str, required=True)
    parser.add_argument("--world", type=str, default="webots_worlds/minimal_lane.wbt")
    parser.add_argument("--round-timesteps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/federated"))
    return parser.parse_args()


def main():
    torch.set_num_threads(max(torch.get_num_threads() // 2, 1))
    args = parse_args()
    client = WebotsFLClient(
        client_id=args.client_id,
        world_path=args.world,
        log_dir=args.log_dir / args.client_id,
        round_timesteps=args.round_timesteps,
        learning_rate=args.learning_rate,
    )
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()

