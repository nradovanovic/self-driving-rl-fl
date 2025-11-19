import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.env_wrapper import make_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the ego car around the ellipse.")
    parser.add_argument("--steps", type=int, default=3000, help="Number of control steps to run.")
    parser.add_argument(
        "--lane-gain",
        type=float,
        default=0.5,
        help="Lane offset gain (higher = tighter centering).",
    )
    parser.add_argument(
        "--heading-gain",
        type=float,
        default=1.0,
        help="Heading alignment gain (higher = faster heading correction).",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.3,
        help="Cruise throttle in [0.1, 1.0] (lower = slower speed).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="How often to print telemetry (steps).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug output every step.",
    )
    parser.add_argument(
        "--lane-only",
        action="store_true",
        help="Use only lane-centering control (no heading/curvature).",
    )
    parser.add_argument(
        "--flip-lane-sign",
        action="store_true",
        help="Flip lane correction sign (for testing if lane offset sign is inverted).",
    )
    parser.add_argument(
        "--debug-file",
        type=Path,
        help="Save debug output to file.",
    )
    return parser.parse_args()


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def autopilot(
    lane_offset: float,
    track_heading: float,
    vehicle_heading: float,
    curvature: float,
    speed_mps: float,
    lane_gain: float,
    heading_gain: float,
    steering_rate: float,
    lane_only: bool = False,
    flip_lane_sign: bool = False,
) -> Tuple[float, dict]:
    """
    Super simple ellipse-following autopilot.
    Strategy: steer back toward center of lane.
    """
    # SIMPLEST: steer toward center of lane
    # If positive lane_offset means right of center, need left steer (negative) → lane_steer = -gain * offset
    # If positive lane_offset means left of center (inverted), need right steer (positive) → lane_steer = gain * offset
    # Since car drifts left but lane_offset is positive, sign seems inverted, so try flipped:
    if flip_lane_sign:
        lane_steer = lane_gain * lane_offset  # Flipped: positive offset (left) → right steer
    else:
        lane_steer = -lane_gain * lane_offset  # Standard: positive offset (right) → left steer
    
    if lane_only:
        steer_cmd = lane_steer
        return float(np.clip(steer_cmd, -1.0, 1.0)), {
            "lane_corr": lane_steer,
            "heading_corr": 0.0,
            "feedforward": 0.0,
            "feedback": lane_steer,
        }
    
    # Also align heading with track heading (secondary, small influence)
    heading_error = wrap_angle(track_heading - vehicle_heading)
    heading_steer = 0.2 * heading_gain * heading_error  # Very small heading correction
    
    # Combine: lane correction is primary (centers car), heading alignment is secondary
    # The lane correction should naturally keep us following the track
    steer_cmd = lane_steer + heading_steer
    
    return float(np.clip(steer_cmd, -1.0, 1.0)), {
        "lane_corr": lane_steer,
        "heading_corr": heading_steer,
        "feedforward": 0.0,
        "feedback": steer_cmd,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WEBOTS_CONTROLLER_URL", "ipc://127.0.0.1:6000/")
    env = make_env(display_progress=True)
    env.config.cruise_throttle = float(np.clip(args.throttle, 0.1, 1.0))
    steering_rate = env.config.steering_rate

    obs, info = env.reset()
    track_heading = info.get("track_heading", 0.0)
    vehicle_heading = info.get("vehicle_heading", 0.0)
    curvature = info.get("track_curvature", 0.0)
    speed_mps = info.get("speed_mps", 0.0)
    
    debug_file = None
    if args.debug_file:
        debug_file = args.debug_file.open("w", encoding="utf-8")
        debug_file.write("# step pos_x pos_z speed lane_off hdg_err track_hdg veh_hdg curv ff fb lane_corr hdg_corr steer reward\n")

    for step in range(args.steps):
        lane_offset = float(obs[2])
        heading_error = wrap_angle(track_heading - vehicle_heading)
        
        steer, control_info = autopilot(
            lane_offset,
            track_heading,
            vehicle_heading,
            curvature,
            speed_mps,
            args.lane_gain,
            args.heading_gain,
            steering_rate,
            args.lane_only,
            args.flip_lane_sign,
        )
        
        action = np.array([steer, 0.0], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        track_heading = info.get("track_heading", track_heading)
        vehicle_heading = info.get("vehicle_heading", vehicle_heading)
        curvature = info.get("track_curvature", curvature)
        speed_mps = info.get("speed_mps", speed_mps)
        
        if args.debug or step % args.log_interval == 0:
            gps_vals = info.get("gps_position", [0, 0, 0])
            msg = (
                f"[step {step:04d}] "
                f"pos=({gps_vals[0]:.3f}, {gps_vals[2]:.3f}) "
                f"speed={speed_mps:.3f} "
                f"lane_off={lane_offset: .3f} "
                f"hdg_err={heading_error:.3f} (track={track_heading:.3f}, veh={vehicle_heading:.3f}) "
                f"curv={curvature:.4f} "
                f"ff={control_info['feedforward']:.3f} "
                f"fb={control_info['feedback']:.3f} (lane={control_info['lane_corr']:.3f} hdg={control_info['heading_corr']:.3f}) "
                f"steer={steer:.3f} "
                f"reward={reward: .3f}"
            )
            print(msg)
            if debug_file:
                debug_file.write(
                    f"{step} {gps_vals[0]:.4f} {gps_vals[2]:.4f} {speed_mps:.4f} "
                    f"{lane_offset:.4f} {heading_error:.4f} {track_heading:.4f} {vehicle_heading:.4f} "
                    f"{curvature:.6f} {control_info['feedforward']:.4f} {control_info['feedback']:.4f} "
                    f"{control_info['lane_corr']:.4f} {control_info['heading_corr']:.4f} {steer:.4f} {reward:.4f}\n"
                )
                debug_file.flush()

        if done or truncated:
            print("Episode finished, resetting simulation...")
            obs, info = env.reset()
            track_heading = info.get("track_heading", 0.0)
            vehicle_heading = info.get("vehicle_heading", 0.0)
            curvature = info.get("track_curvature", 0.0)
            speed_mps = info.get("speed_mps", 0.0)

    if debug_file:
        debug_file.close()
    
    # Properly close environment and give Webots time to clean up
    try:
        env.close()
    except Exception:
        pass  # Ignore errors during cleanup
    finally:
        # Small delay to let Webots release resources
        time.sleep(0.1)


if __name__ == "__main__":
    main()

