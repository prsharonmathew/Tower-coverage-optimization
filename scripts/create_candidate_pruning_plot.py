from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tower_placement.data_loader import CityDataLoader
from tower_placement.candidate_geometry import CandidateTowerGenerator, FeasiblePairBuilder


DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "cities" / "cities_de_50k.txt"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "results" / "figures" / "candidate_pruning_radius_18_48.png"
EARTH_RADIUS_KM = 6371.0
GRID_STEP = 0.02
GRID_MARGIN = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot candidate tower pruning for a radius pair."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="City coordinate file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="PNG output path.",
    )
    parser.add_argument(
        "--radius-pair",
        type=float,
        nargs=2,
        default=(18.0, 48.0),
        metavar=("R1", "R2"),
        help="Radius pair in km. Pruning uses the larger value.",
    )
    return parser.parse_args()


def split_kept_and_pruned(
    towers: pd.DataFrame,
    feasible_pairs_by_city: dict[int, list[tuple[int, float]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    used_tower_indices = {
        int(tower_index)
        for feasible_pairs in feasible_pairs_by_city.values()
        for tower_index, _ in feasible_pairs
    }

    keep_mask = towers.index.to_series().isin(used_tower_indices)
    kept_towers = towers.loc[keep_mask].copy()
    pruned_towers = towers.loc[~keep_mask].copy()
    return kept_towers, pruned_towers


def plot_pruning(
    cities: pd.DataFrame,
    kept_towers: pd.DataFrame,
    pruned_towers: pd.DataFrame,
    radius_pair: Tuple[float, float],
    output_file: Path,
) -> None:
    radius_small, radius_large = sorted(float(radius) for radius in radius_pair)
    total_towers = len(kept_towers) + len(pruned_towers)

    fig, ax = plt.subplots(figsize=(12, 8.5))

    if not pruned_towers.empty:
        ax.scatter(
            pruned_towers["longitude"],
            pruned_towers["latitude"],
            s=4,
            color="#dc2626",
            alpha=0.28,
            linewidths=0,
            label=f"Pruned candidate towers: {len(pruned_towers):,} removed",
            zorder=1,
        )

    if not kept_towers.empty:
        ax.scatter(
            kept_towers["longitude"],
            kept_towers["latitude"],
            s=4,
            color="#2563eb",
            alpha=0.42,
            linewidths=0,
            label=f"Kept candidate towers: {len(kept_towers):,}",
            zorder=2,
        )

    ax.scatter(
        cities["longitude"],
        cities["latitude"],
        s=16,
        color="#111111",
        alpha=0.95,
        linewidths=0,
        label=f"Cities: {len(cities):,}",
        zorder=3,
    )

    ax.set_title(
        "Candidate Tower Pruning"
        f" | Radius pair ({radius_small:g} km, {radius_large:g} km)"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}N"))

    mean_latitude = float(cities["latitude"].mean())
    ax.set_aspect(1 / math.cos(math.radians(mean_latitude)))

    ax.set_xlim(
        float(cities["longitude"].min()) - 0.5,
        float(cities["longitude"].max()) + 0.5,
    )
    ax.set_ylim(
        float(cities["latitude"].min()) - 0.5,
        float(cities["latitude"].max()) + 0.5,
    )

    ax.grid(True, which="major", linestyle="--", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.12)

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        title=f"Pruning cutoff: {radius_large:g} km",
    )
    legend._legend_box.align = "left"

    stats_text = (
        f"All candidates: {total_towers:,}\n"
        f"Removed by pruning: {len(pruned_towers):,}\n"
        f"Kept for MILP: {len(kept_towers):,}"
    )
    ax.text(
        1.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92},
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(output_file, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    radius_pair = tuple(float(radius) for radius in args.radius_pair)
    radius_large = max(radius_pair)

    cities = CityDataLoader().load(args.input_file)
    towers = CandidateTowerGenerator().generate(
        cities,
        grid_step=GRID_STEP,
        grid_margin=GRID_MARGIN,
    )

    feasible_pairs_by_city = FeasiblePairBuilder(EARTH_RADIUS_KM).build(
        cities,
        towers,
        max_radius_km=radius_large,
    )
    kept_towers, pruned_towers = split_kept_and_pruned(towers, feasible_pairs_by_city)

    plot_pruning(
        cities=cities,
        kept_towers=kept_towers,
        pruned_towers=pruned_towers,
        radius_pair=radius_pair,
        output_file=args.output_file,
    )

    print(f"Cities: {len(cities):,}")
    print(f"Candidate towers before pruning: {len(towers):,}")
    print(f"Candidate towers kept: {len(kept_towers):,}")
    print(f"Candidate towers removed by pruning: {len(pruned_towers):,}")
    print(f"Pruning cutoff radius: {radius_large:g} km")
    print(f"Saved plot: {args.output_file}")


if __name__ == "__main__":
    main()
