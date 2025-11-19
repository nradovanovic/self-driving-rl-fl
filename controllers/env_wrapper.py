import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

def _extend_controller_path() -> None:
    """Add Webots controller Python path automatically when possible."""
    candidate_paths = []
    env_home = os.environ.get("WEBOTS_HOME")
    if env_home:
        candidate_paths.append(Path(env_home) / "lib" / "controller" / "python")
    if os.name == "nt":
        candidate_paths.append(Path("C:/Program Files/Webots/lib/controller/python"))
    else:
        candidate_paths.append(Path("/usr/local/webots/lib/controller/python"))
    for path in candidate_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            if "WEBOTS_HOME" not in os.environ:
                # Backfill WEBOTS_HOME so the native library loader works.
                potential_home = Path(path).parents[3]
                os.environ["WEBOTS_HOME"] = str(potential_home)
                os.environ.setdefault("WEBOTS_CONTROLLER_PATH", str(path))
                os.environ.setdefault(
                    "WEBOTS_CONTROLLER_LIB_PATH",
                    str(Path(path).parents[2] / "controller"),
                )
            break


_extend_controller_path()

try:
    from controller import Supervisor  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    Supervisor = None  # type: ignore


class WebotsNotAvailableError(RuntimeError):
    """Raised when the Webots controller module cannot be loaded."""


@dataclass
class EnvConfig:
    max_speed: float = 4.0  # m/s
    accel: float = 0.5
    steering_rate: float = math.radians(180)  # rad/s, tuned for 32 ms timestep
    lane_half_width: float = 1.5
    world_length: float = 18.0
    collision_distance: float = 0.2
    episode_duration: float = 30.0  # seconds
    cruise_throttle: float = 0.6
    track_shape: str = "ellipse"
    ellipse_center_a: float = 5.25
    ellipse_center_b: float = 2.25
    ellipse_width_ratio: float = 0.14285714285714285
    start_translation: Tuple[float, float, float] = (0.0, 0.05, -2.25)
    start_heading: float = 0.0
    cruise_throttle: float = 0.6  # constant forward command in [0, 1]
    max_track_curvature: float = 1.2  # used for observation normalization


class WebotsDrivingEnv(gym.Env):
    """Gym-compatible wrapper around a minimal Webots world."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        time_step_ms: Optional[int] = None,
        config: EnvConfig = EnvConfig(),
        auto_connect: bool = True,
        display_progress: bool = False,
    ) -> None:
        super().__init__()
        if Supervisor is None:
            raise WebotsNotAvailableError(
                "The Webots Python API could not be imported. "
                "Install Webots and run this environment as an extern controller."
            )

        self.config = config
        self.display_progress = display_progress
        self._debug = display_progress  # Enable debug prints if showing progress
        self.supervisor: Optional[Supervisor] = None
        self.robot = None
        self.front_sensor = None
        self.left_sensor = None
        self.right_sensor = None
        self.gps = None
        self.time_step = time_step_ms
        self._prev_position = np.zeros(3, dtype=np.float32)
        self._speed = 0.0
        self._heading = 0.0
        self._last_forward_progress = 0.0
        self._last_tangent = np.array([1.0, 0.0], dtype=np.float32)
        self._last_theta = 0.0
        self._last_curvature = 0.0
        self._elapsed = 0.0
        self._episode_steps = 0
        self._max_steps = int(
            (self.config.episode_duration * 1000) / (self.time_step or 32)
        )

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.5, -1.0, -1.5, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        if auto_connect:
            self._connect()

    def _connect(self) -> None:
        self.supervisor = Supervisor()
        if self.supervisor is None:
            raise RuntimeError("Unable to instantiate Webots Supervisor. Is Webots running?")

        if self.time_step is None:
            self.time_step = int(self.supervisor.getBasicTimeStep())
        self._max_steps = int((self.config.episode_duration * 1000) / self.time_step)

        self.robot = self.supervisor.getSelf()
        if self.robot is None:
            raise RuntimeError("Supervisor could not access self robot.")

        self.front_sensor = self.supervisor.getDevice("front_sensor")
        self.left_sensor = self.supervisor.getDevice("left_sensor")
        self.right_sensor = self.supervisor.getDevice("right_sensor")
        self.gps = self.supervisor.getDevice("gps")
        for sensor in (self.front_sensor, self.left_sensor, self.right_sensor, self.gps):
            if sensor is None:
                raise RuntimeError("Missing sensor device in Webots world.")
            sensor.enable(self.time_step)

        if self.display_progress:
            print("Connected to Webots world (timestep=%s ms)" % self.time_step)

    # Gym API -----------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if self.supervisor is None:
            self._connect()

        translation_field = self.robot.getField("translation")  # type: ignore
        rotation_field = self.robot.getField("rotation")  # type: ignore

        translation_field.setSFVec3f(list(self.config.start_translation))
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(self.time_step)

        self._speed = 0.0
        self._elapsed = 0.0
        self._episode_steps = 0
        self._prev_position = np.array(self.gps.getValues(), dtype=np.float32)
        _, tangent, theta, curvature = self._compute_track_metrics(self._prev_position)
        self._last_tangent = tangent
        self._last_theta = theta
        self._last_curvature = curvature
        self._last_forward_progress = 0.0
        
        # Align vehicle heading with track heading at start
        track_heading = math.atan2(tangent[1], tangent[0])
        self._heading = self._wrap_angle(track_heading)
        rotation_field.setSFRotation([0.0, 1.0, 0.0, self._heading])
        self.supervisor.step(self.time_step)

        obs = self._get_observation()
        info = {
            "reset_time": time.time(),
            "track_heading": self._get_track_heading(),
            "vehicle_heading": self._heading,
        }
        return obs, info

    def step(self, action: np.ndarray):
        steering_input = float(np.clip(action[0], -1.0, 1.0))
        steer = steering_input
        throttle = np.clip(self.config.cruise_throttle, 0.1, 1.0)
        dt = self.time_step / 1000.0

        # Speed tracking: accelerate toward target cruise speed
        target_speed = throttle * self.config.max_speed
        speed_error = target_speed - self._speed
        
        # Use aggressive acceleration (proportional with max limit)
        # This ensures speed increases quickly instead of staying stuck at tiny values
        accel_command = np.clip(speed_error * 10.0, -self.config.accel, self.config.accel)
        self._speed += accel_command * dt
        
        # Cap at max speed
        self._speed = float(np.clip(self._speed, 0.0, self.config.max_speed))
        
        # CRITICAL FIX: Force minimum speed if we're supposed to be moving
        # Prevents getting stuck at tiny values due to numerical issues
        if target_speed > 0.5 and self._speed < 0.5:
            # If target is high but speed is low, jump-start it
            self._speed = max(self._speed, 0.5)
        elif target_speed > 0.1 and self._speed < 0.1:
            self._speed = 0.1

        # update heading using steering rate
        # Standard: positive steer = right turn (increase heading), negative = left turn (decrease)
        self._heading += steer * self.config.steering_rate * dt
        self._heading = self._wrap_angle(self._heading)

        dx = self._speed * math.cos(self._heading) * dt
        dz = self._speed * math.sin(self._heading) * dt

        pose_field = self.robot.getField("translation")  # type: ignore
        rot_field = self.robot.getField("rotation")  # type: ignore
        position = np.array(pose_field.getSFVec3f(), dtype=np.float32)
        position += np.array([dx, 0.0, dz], dtype=np.float32)
        pose_field.setSFVec3f(position.tolist())
        rot_field.setSFRotation([0.0, 1.0, 0.0, self._heading])
        
        # Step physics - position should now be applied
        self.supervisor.step(self.time_step)
        
        # Verify position was set (if physics fights it, this helps debug)
        # Force position again if needed (some Webots versions need this)
        if self._speed > 0.05:
            actual_pos = np.array(pose_field.getSFVec3f(), dtype=np.float32)
            expected_pos = position
            if np.linalg.norm(actual_pos - expected_pos) > 0.01:
                # Position was reset, force it again
                pose_field.setSFVec3f(position.tolist())
        self._elapsed += dt
        self._episode_steps += 1

        obs = self._get_observation()
        reward, info = self._get_reward(obs, steer)
        terminated = self._is_done(obs)
        truncated = self._episode_steps >= self._max_steps
        return obs, reward, terminated, truncated, info

    # Helpers -----------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        gps_vals = np.array(self.gps.getValues(), dtype=np.float32)
        delta = gps_vals - self._prev_position
        self._prev_position = gps_vals.copy()
        dt = self.time_step / 1000.0
        lane_offset, tangent, theta, curvature = self._compute_track_metrics(gps_vals)
        displacement = np.array([delta[0], delta[2]], dtype=np.float32)
        forward = max(0.0, float(np.dot(displacement, tangent)))
        denom = max(self.config.max_speed * dt, 1e-4)
        self._last_forward_progress = forward / denom
        self._last_tangent = tangent
        self._last_theta = theta
        self._last_curvature = curvature
        speed = float(np.linalg.norm(delta) / max(dt, 1e-4))

        front = float(self.front_sensor.getValue())
        left = float(self.left_sensor.getValue())
        right = float(self.right_sensor.getValue())

        max_sensor = 5.0
        track_heading = np.clip(self._last_theta / math.pi, -1.0, 1.0)
        curvature_norm = np.clip(
            self._last_curvature / self.config.max_track_curvature, -1.0, 1.0
        )
        obs = np.array(
            [
                np.clip(speed / self.config.max_speed, 0.0, 1.0),
                np.clip(self._heading / math.pi, -1.0, 1.0),
                lane_offset,
                track_heading,
                curvature_norm,
                np.clip(front / max_sensor, 0.0, 1.0),
                np.clip(left / max_sensor, 0.0, 1.0),
                np.clip(right / max_sensor, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_track_metrics(
        self, position: np.ndarray
    ) -> Tuple[float, np.ndarray, float, float]:
        x = float(position[0])
        z = float(position[2])
        if self.config.track_shape == "ellipse":
            a = self.config.ellipse_center_a
            b = self.config.ellipse_center_b
            width_ratio = max(self.config.ellipse_width_ratio, 1e-4)
            
            # Compute theta from normalized coordinates
            theta = math.atan2((z / b), (x / a))
            
            # Centerline point at this theta
            centerline_x = a * math.cos(theta)
            centerline_z = b * math.sin(theta)
            
            # Tangent direction (derivative of ellipse)
            dx = -a * math.sin(theta)
            dz = b * math.cos(theta)
            tangent = np.array([dx, dz], dtype=np.float32)
            tangent_norm = tangent / max(float(np.linalg.norm(tangent)), 1e-4)
            
            # Normal direction (perpendicular to tangent, pointing outward from ellipse center)
            # Tangent = [-a*sin(theta), b*cos(theta)] points counterclockwise
            # Outward normal: for ellipse (x/a)² + (z/b)² = 1, outward is away from center (0,0)
            # At point (a*cos(theta), b*sin(theta)), outward direction is [b*cos(theta), a*sin(theta)]
            # This is perpendicular to tangent and points outward:
            # At theta=0 (rightmost): [b, 0] points right (outward) ✓
            # At theta=-π/2 (bottom): [0, -a] points down (outward from center) ✓
            normal = np.array([b * math.cos(theta), a * math.sin(theta)], dtype=np.float32)
            normal_norm = normal / max(float(np.linalg.norm(normal)), 1e-4)
            
            # Vector from centerline to vehicle
            vec_to_vehicle = np.array([x - centerline_x, z - centerline_z], dtype=np.float32)
            
            # Signed distance along normal: positive = outside ellipse (right), negative = inside (left)
            signed_distance = float(np.dot(vec_to_vehicle, normal_norm))
            lane_offset = float(np.clip(signed_distance / (width_ratio * a), -1.5, 1.5))
            
            # Curvature computation
            ddx = -a * math.cos(theta)
            ddz = -b * math.sin(theta)
            numerator = dx * ddz - dz * ddx
            denominator = (dx * dx + dz * dz) ** 1.5
            curvature = numerator / max(denominator, 1e-6)
        else:
            lane_offset = float(np.clip(z / self.config.lane_half_width, -1.5, 1.5))
            tangent = np.array([1.0, 0.0], dtype=np.float32)
            tangent_norm = tangent
            theta = 0.0
            curvature = 0.0
        return lane_offset, tangent_norm, theta, curvature

    def _get_track_heading(self) -> float:
        return float(math.atan2(self._last_tangent[1], self._last_tangent[0]))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + math.pi) % (2 * math.pi) - math.pi)

    def _get_reward(self, obs: np.ndarray, steer: float) -> Tuple[float, Dict]:
        lane_penalty = -abs(obs[2]) * 0.8
        speed_reward = obs[0]
        progress_reward = self._last_forward_progress * 1.5
        steering_penalty = -abs(steer) * 0.1
        front_distance = obs[5]
        collision_penalty = (
            -1.0 if front_distance < (self.config.collision_distance / 5.0) else 0.0
        )
        reward = (
            speed_reward
            + progress_reward
            + lane_penalty
            + steering_penalty
            + collision_penalty
        )
        gps_vals = np.array(self.gps.getValues(), dtype=np.float32)
        info = {
            "speed": float(obs[0]),
            "speed_mps": self._speed,
            "lane_offset": float(obs[2]),
            "front_distance": float(front_distance),
            "collision": collision_penalty < 0,
            "track_heading": self._get_track_heading(),
            "vehicle_heading": self._heading,
            "track_curvature": self._last_curvature,
            "gps_position": gps_vals.tolist(),
        }
        return reward, info

    def _is_done(self, obs: np.ndarray) -> bool:
        out_of_lane = abs(obs[2]) > 1.0
        collision = obs[5] < (self.config.collision_distance / 5.0)
        reached_end = self._elapsed >= self.config.episode_duration
        return bool(out_of_lane or collision or reached_end)

    def render(self, mode="human"):
        return None

    def close(self):
        if self.supervisor:
            self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            self.supervisor = None


def make_env(**kwargs) -> WebotsDrivingEnv:
    """Factory helper."""
    return WebotsDrivingEnv(**kwargs)


if __name__ == "__main__":
    env = WebotsDrivingEnv(auto_connect=True, display_progress=True)
    obs, _ = env.reset()
    print("Initial observation:", obs)
    for _ in range(10):
        sample_action = env.action_space.sample()
        obs, reward, done, _, info = env.step(sample_action)
        print("Reward:", reward, "Info:", info)
        if done:
            env.reset()

