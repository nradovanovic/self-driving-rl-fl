import argparse
from pathlib import Path

import flwr as fl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for Webots RL FL.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-fit-clients", type=int, default=2)
    parser.add_argument("--min-available-clients", type=int, default=2)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/federated/server"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "reward": sum(m.get("reward", 0.0) for m in metrics) / max(len(metrics), 1),
            "collision_rate": sum(m.get("collision_rate", 0.0) for m in metrics)
            / max(len(metrics), 1),
        },
    )

    fl.server.start_server(
        server_address="[::]:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()

