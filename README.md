# Self-Driving RL with Federated Learning in Webots

This project trains lightweight autonomous driving agents in Webots using Stable-Baselines3 (PPO) and aggregates multiple clients via Flower-powered Federated Learning. The entire stack is optimized for CPU-only, headless execution (Xvfb) so it can run on modest hardware without dedicated GPUs.

## Project Layout

```
project_root/
├── controllers/
│   ├── env_wrapper.py
│   ├── federated_client.py
│   └── rl_training.py
├── docs/
│   ├── run_instructions.md
│   └── setup_guide.md
├── metrics/
│   ├── evaluation.py
│   └── plots.py
├── server/
│   └── flower_server.py
├── webots_worlds/
│   └── minimal_lane.wbt
├── requirements.txt
└── README.md
```

## Quick Start

1. **Install prerequisites**
   - Python 3.10+
   - Webots R2024b or newer
   - Linux (Ubuntu) users: install `xvfb` for headless mode.
     ```
     sudo apt update
     sudo apt install xvfb python3-venv
     ```
   - Windows users: no extra system packages are needed; install Webots for Windows (default `C:\Program Files\Webots`) and set the environment variables shown below.
2. **Create virtual environment**
   - Linux/macOS:
     ```
     python -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - Windows (PowerShell):
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     python.exe -m pip install --upgrade pip
     pip install -r requirements.txt
     ```
   PyTorch installs in CPU-only mode automatically via PyPI wheels specified in `requirements.txt`.
3. **Headless Webots session**
   - Linux (Xvfb):
     ```
     Xvfb :0 -screen 0 1024x768x24 &
     export DISPLAY=:0
     export WEBOTS_HOME=/usr/local/webots        # adjust if custom install
     export WEBOTS_CONTROLLER_URL=ipc://127.0.0.1:6000/
     webots --stdout --stderr --mode=fast --no-rendering \
       webots_worlds/minimal_lane.wbt &
     ```
   - Windows (PowerShell):
     ```
     setx WEBOTS_HOME "C:\Program Files\Webots"
     $env:WEBOTS_HOME = "C:\Program Files\Webots"
     setx WEBOTS_CONTROLLER_URL "tcp://127.0.0.1:1234/EGO_CAR"
     $env:WEBOTS_CONTROLLER_URL = "tcp://127.0.0.1:1234/EGO_CAR"
     & "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
        --mode=fast --no-rendering --stdout --stderr `
        webots_worlds\minimal_lane.wbt
     ```
     **Important**: When Webots starts, check the console output for the controller URL (e.g., `Controller: extern (ipc://127.0.0.1:6000/)` or `Waiting on port 1234 targeting 'EGO_CAR'`). You must set `WEBOTS_CONTROLLER_URL` in every terminal running Python scripts to match this exact value, otherwise the controllers won't connect.
4. **Run a standalone local RL session**
   ```
   python controllers/rl_training.py --timesteps 2000 --log-dir runs/client1
   ```
5. **Launch Flower federated training**
   - Server: `python server/flower_server.py --rounds 5`
   - Client(s):
     ```
     python controllers/federated_client.py \
       --client-id client1 \
       --world webots_worlds/minimal_lane.wbt \
       --round-timesteps 1000
     ```
   Spawn multiple clients (each in its own terminal) pointing to the same Flower server to emulate decentralized training.
6. **Inspect metrics and plots**
   ```
   # Evaluate metrics
   python metrics/evaluation.py --metrics-file runs/client1/metrics.csv
   
   # Plot individual metrics
   python metrics/plots.py --metrics-file runs/local_client/metrics.csv --output-dir metrics/figures
   python metrics/plots.py --metrics-file runs/federated/client01/metrics.csv --output-dir metrics/figures
   
   # Compare local training vs federated learning side-by-side
   python metrics/compare_plots.py \
     --local-metrics runs/local_client/metrics.csv \
     --federated-metrics runs/federated/client01/metrics.csv \
     --output-dir metrics/figures
   ```

## CPU-Only & Resource Efficiency

- Stable-Baselines3 models use compact MLP policies with small batch sizes.
- Webots world avoids camera sensors, relying on velocities, steering angle, and distance sensors only.
- Controller timestep and number of RL steps per federated round default to conservative values to keep resource usage minimal.

## Metrics

Training scripts log episode reward, collision counts, lane deviation, and federated round aggregates to CSV files under `runs/`. The plotting utility generates PNG charts comparing local and global performance.

### Understanding the Metrics

**Local Training Metrics** (`runs/local_client/metrics.csv`):
- **X-axis**: Training steps (one data point per episode)
- **Content**: Training metrics logged during each episode
- Shows learning progress as the agent trains

**Federated Learning Metrics** (`runs/federated/{client_id}/metrics.csv`):
- **X-axis**: Federated learning rounds (one data point per round)
- **Content**: Evaluation metrics after each round (average of 3 evaluation episodes)
- Shows how well the aggregated model performs when evaluated
- Individual training episodes within each round are not logged

**Key Difference**: Local training shows per-episode training progress, while federated learning shows per-round evaluation of the aggregated model. The comparison plots normalize both to "Training Progress (%)" for fair comparison.

