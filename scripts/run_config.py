"""Runs the experiment as configured by the specified configuration file."""


import argparse

from defame.eval.utils import run_experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run an experiment based on the provided configuration file."
    )
    parser.add_argument(
        "config_file_path",
        type=str,
        default="/DEFAME/config/experiments/averitec.yaml",
        help="Path to the configuration file for the experiment."
    )
    args = parser.parse_args()

    print(f"Running experiment with config file: {args.config_file_path}")
    run_experiment(config_file_path=args.config_file_path)
