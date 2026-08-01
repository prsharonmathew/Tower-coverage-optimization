from __future__ import annotations

import argparse
from pathlib import Path

from tower_placement.app import INSTANCE_CONFIGS, RadiusSearchProgram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the modular tower-placement radius search."
    )
    parser.add_argument(
        "--instance",
        choices=sorted(INSTANCE_CONFIGS),
        default="france",
        help="Country instance to solve.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to results/search_runs/<instance>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RadiusSearchProgram(
        instance_name=args.instance,
        output_dir=args.output_dir,
    ).run()


if __name__ == "__main__":
    main()
