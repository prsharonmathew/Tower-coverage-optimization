from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CITY_FILE = PROJECT_ROOT / "data" / "cities" / "cities_de_50k.txt"
SELECTED_TOWERS_FILE = PROJECT_ROOT / "results" / "selected_towers_germany.csv"
OUTPUT_DIR = PROJECT_ROOT / "validation" / "independent_germany"

LAMBDA_PENALTY = 2.0
TOWER_COSTS: Dict[float, float] = {
    17.8: 0.9548110056672545,
    18.0: 0.958283,
    47.9: 2.2526929866666667,
    48.0: 2.259293,
}


def load_city_data(file_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
        encoding="utf-8",
        skipinitialspace=True,
        header=None,
        names=["city", "latitude", "longitude"],
    )

    city_ids = raw_df["city"].astype(str).str.strip()
    city_ids = city_ids.where(city_ids != "", raw_df.index.astype(str))

    cities = pd.DataFrame(
        {
            "city_id": city_ids,
            "latitude": pd.to_numeric(raw_df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw_df["longitude"], errors="coerce"),
        }
    ).reset_index(drop=True)

    if cities["latitude"].isna().any() or cities["longitude"].isna().any():
        bad_rows = cities[cities["latitude"].isna() | cities["longitude"].isna()]
        raise ValueError(f"Invalid city coordinates at rows: {bad_rows.index.tolist()}")

    return cities


def load_selected_towers(file_path: Path) -> pd.DataFrame:
    towers = pd.read_csv(file_path)
    required_columns = {"latitude", "longitude", "radius"}
    missing_columns = required_columns - set(towers.columns)
    if missing_columns:
        raise ValueError(f"Selected tower file is missing columns: {missing_columns}")

    towers = towers[["latitude", "longitude", "radius"]].copy()
    towers["tower_id"] = range(len(towers))
    towers["latitude"] = pd.to_numeric(towers["latitude"], errors="coerce")
    towers["longitude"] = pd.to_numeric(towers["longitude"], errors="coerce")
    towers["radius"] = pd.to_numeric(towers["radius"], errors="coerce")

    if towers[["latitude", "longitude", "radius"]].isna().any().any():
        raise ValueError("Selected tower file contains invalid numeric values.")

    return towers


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c


def tower_cost(radius: float) -> float:
    radius_key = float(radius)
    if radius_key not in TOWER_COSTS:
        raise ValueError(f"No tower cost configured for radius {radius_key}")
    return TOWER_COSTS[radius_key]


def verify_solution(cities: pd.DataFrame, towers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    city_rows: List[dict] = []
    tower_coverage_counts = {int(row["tower_id"]): 0 for _, row in towers.iterrows()}

    for _, city in cities.iterrows():
        covering_tower_ids: List[int] = []
        covering_radii: List[float] = []
        nearest_tower_id = None
        nearest_distance = float("inf")

        for _, tower in towers.iterrows():
            tower_id = int(tower["tower_id"])
            distance = haversine_km(
                float(city["latitude"]),
                float(city["longitude"]),
                float(tower["latitude"]),
                float(tower["longitude"]),
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_tower_id = tower_id

            if distance <= float(tower["radius"]):
                covering_tower_ids.append(tower_id)
                covering_radii.append(float(tower["radius"]))
                tower_coverage_counts[tower_id] += 1

        coverage_count = len(covering_tower_ids)
        city_rows.append(
            {
                "city_id": city["city_id"],
                "city_latitude": float(city["latitude"]),
                "city_longitude": float(city["longitude"]),
                "coverage_count": coverage_count,
                "is_covered": coverage_count >= 1,
                "has_interference": coverage_count > 1,
                "excess_coverage": max(0, coverage_count - 1),
                "covering_tower_ids": ";".join(str(tower_id) for tower_id in covering_tower_ids),
                "covering_radii": ";".join(f"{radius:g}" for radius in covering_radii),
                "nearest_tower_id": nearest_tower_id,
                "nearest_tower_distance_km": nearest_distance,
            }
        )

    city_coverage = pd.DataFrame(city_rows)

    tower_rows = []
    for _, tower in towers.iterrows():
        radius = float(tower["radius"])
        tower_rows.append(
            {
                "tower_id": int(tower["tower_id"]),
                "latitude": float(tower["latitude"]),
                "longitude": float(tower["longitude"]),
                "radius": radius,
                "installation_cost": tower_cost(radius),
                "covered_city_count": tower_coverage_counts[int(tower["tower_id"])],
            }
        )
    tower_summary = pd.DataFrame(tower_rows)

    total_excess_coverage = int(city_coverage["excess_coverage"].sum())
    interference_penalty = LAMBDA_PENALTY * total_excess_coverage
    installation_cost = float(tower_summary["installation_cost"].sum())
    objective_value = installation_cost + interference_penalty

    summary = {
        "city_file": str(CITY_FILE),
        "selected_towers_file": str(SELECTED_TOWERS_FILE),
        "num_cities": int(len(cities)),
        "num_selected_towers": int(len(towers)),
        "num_uncovered_cities": int((~city_coverage["is_covered"]).sum()),
        "num_cities_covered_exactly_once": int((city_coverage["coverage_count"] == 1).sum()),
        "num_cities_with_interference": int((city_coverage["coverage_count"] > 1).sum()),
        "max_coverage_count": int(city_coverage["coverage_count"].max()),
        "total_excess_coverage": total_excess_coverage,
        "lambda_penalty": LAMBDA_PENALTY,
        "interference_penalty": interference_penalty,
        "installation_cost": installation_cost,
        "objective_value": objective_value,
        "tower_count_by_radius": (
            tower_summary.groupby("radius")["tower_id"].count().to_dict()
        ),
        "cost_by_radius": (
            tower_summary.groupby("radius")["installation_cost"].sum().to_dict()
        ),
    }

    return city_coverage, tower_summary, summary


def write_summary(summary: dict, output_file: Path) -> None:
    lines = [
        "Independent Selected Tower Verification",
        "=======================================",
        "",
        f"City file: {summary['city_file']}",
        f"Selected towers file: {summary['selected_towers_file']}",
        "",
        f"Number of cities: {summary['num_cities']}",
        f"Number of selected towers: {summary['num_selected_towers']}",
        f"Uncovered cities: {summary['num_uncovered_cities']}",
        f"Cities covered exactly once: {summary['num_cities_covered_exactly_once']}",
        f"Cities with interference: {summary['num_cities_with_interference']}",
        f"Maximum coverage count for any city: {summary['max_coverage_count']}",
        f"Total excess coverage: {summary['total_excess_coverage']}",
        "",
        f"Lambda: {summary['lambda_penalty']}",
        f"Installation cost: {summary['installation_cost']:.6f}",
        f"Interference penalty: {summary['interference_penalty']:.6f}",
        f"Independent objective value: {summary['objective_value']:.6f}",
        "",
        f"Tower count by radius: {summary['tower_count_by_radius']}",
        f"Cost by radius: {summary['cost_by_radius']}",
    ]
    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cities = load_city_data(CITY_FILE)
    towers = load_selected_towers(SELECTED_TOWERS_FILE)
    city_coverage, tower_summary, summary = verify_solution(cities, towers)

    city_coverage_file = OUTPUT_DIR / "independent_city_coverage_verification.csv"
    tower_summary_file = OUTPUT_DIR / "independent_tower_coverage_summary.csv"
    summary_csv_file = OUTPUT_DIR / "independent_solution_verification_summary.csv"
    summary_txt_file = OUTPUT_DIR / "independent_solution_verification_summary.txt"

    city_coverage.to_csv(city_coverage_file, index=False)
    tower_summary.to_csv(tower_summary_file, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv_file, index=False)
    write_summary(summary, summary_txt_file)

    print(summary_txt_file.read_text(encoding="utf-8"))
    print("")
    print(f"City coverage details: {city_coverage_file}")
    print(f"Tower coverage summary: {tower_summary_file}")
    print(f"Summary CSV: {summary_csv_file}")
    print(f"Summary TXT: {summary_txt_file}")


if __name__ == "__main__":
    main()
