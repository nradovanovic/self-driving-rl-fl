"""
Path logger callback to track car positions during training episodes.
Saves paths for visualization of learning progress.
"""
import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PathLogger(BaseCallback):
    """Logs car positions during episodes for path visualization."""
    
    def __init__(self, paths_dir: Path, log_every_n_episodes: int = 10, verbose: int = 0):
        """
        Args:
            paths_dir: Directory to save path JSON files
            log_every_n_episodes: Log path every N episodes (to avoid too many files)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.paths_dir = paths_dir
        self.paths_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_episodes = log_every_n_episodes
        self.episode_count = 0
        self.current_episode_path: List[Tuple[float, float]] = []
        self.current_episode_reward = 0.0
        self.current_episode_collision = False
        
    def _on_step(self) -> bool:
        """Track positions during each step."""
        infos = self.locals.get("infos", [])
        for info in infos:
            # Track position during episode (always collect, even if we won't save)
            gps_position = info.get("gps_position")
            if gps_position is not None:
                # Store (x, z) coordinates (y is height, ignore it)
                self.current_episode_path.append((float(gps_position[0]), float(gps_position[2])))
            
            # Track collision
            if info.get("collision", False):
                self.current_episode_collision = True
            
            # Check if episode ended
            if info.get("episode"):
                self.episode_count += 1
                # Get final episode reward
                self.current_episode_reward = float(info["episode"].get("r", 0.0))
                
                # Save path if it's time to log
                if (self.episode_count - 1) % self.log_every_n_episodes == 0:
                    self._save_path()
                
                # Reset for next episode
                self.current_episode_path = []
                self.current_episode_reward = 0.0
                self.current_episode_collision = False
                    
        return True
    
    def _save_path(self) -> None:
        """Save current episode path to JSON file."""
        if len(self.current_episode_path) == 0:
            return
            
        path_data = {
            "episode": self.episode_count,
            "reward": self.current_episode_reward,
            "collision": self.current_episode_collision,
            "path": self.current_episode_path,
        }
        
        path_file = self.paths_dir / f"episode_{self.episode_count:05d}.json"
        with path_file.open("w") as f:
            json.dump(path_data, f, indent=2)

