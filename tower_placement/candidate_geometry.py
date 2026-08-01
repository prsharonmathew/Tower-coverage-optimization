from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


class CandidateTowerGenerator:
    def generate(
        self,
        cities: pd.DataFrame,
        grid_step: float,
        grid_margin: float,
    ) -> pd.DataFrame:
        min_lat = float(cities["latitude"].min()) - grid_margin
        max_lat = float(cities["latitude"].max()) + grid_margin
        min_lon = float(cities["longitude"].min()) - grid_margin
        max_lon = float(cities["longitude"].max()) + grid_margin

        candidate_list: List[Tuple[float, float]] = []
        current_lat = min_lat
        while current_lat <= max_lat + 1e-12:
            current_lon = min_lon
            while current_lon <= max_lon + 1e-12:
                candidate_list.append((current_lat, current_lon))
                current_lon += grid_step
            current_lat += grid_step

        tower_data = pd.DataFrame(candidate_list, columns=["latitude", "longitude"])
        tower_data["tower_id"] = range(len(tower_data))
        return tower_data


class FeasiblePairBuilder:
    def __init__(self, earth_radius_km: float) -> None:
        self.earth_radius_km = earth_radius_km

    def build(
        self,
        cities: pd.DataFrame,
        towers: pd.DataFrame,
        max_radius_km: float,
    ) -> Dict[int, List[Tuple[int, float]]]:
        feasible_pairs_by_city: Dict[int, List[Tuple[int, float]]] = {
            int(i): [] for i in cities.index
        }
        if cities.empty or towers.empty:
            return feasible_pairs_by_city

        city_coords_rad = np.radians(cities[["latitude", "longitude"]].to_numpy(dtype=float))
        tower_coords_rad = np.radians(towers[["latitude", "longitude"]].to_numpy(dtype=float))
        query_radius = float(max_radius_km) / self.earth_radius_km
        tower_tree = BallTree(tower_coords_rad, metric="haversine")

        index_lists, distance_lists = tower_tree.query_radius(
            city_coords_rad,
            r=query_radius,
            return_distance=True,
            sort_results=True,
        )

        for city_position, city_index in enumerate(cities.index):
            city_pairs = []
            for tower_index, distance_rad in zip(
                index_lists[city_position],
                distance_lists[city_position],
            ):
                city_pairs.append((int(tower_index), float(distance_rad * self.earth_radius_km)))
            feasible_pairs_by_city[int(city_index)] = city_pairs

        return feasible_pairs_by_city


class TowerPruner:
    def prune(
        self,
        towers: pd.DataFrame,
        feasible_pairs_by_city: Dict[int, List[Tuple[int, float]]],
    ) -> Tuple[pd.DataFrame, Dict[int, List[Tuple[int, float]]], int]:
        used_tower_indices = sorted(
            {
                int(tower_index)
                for feasible_pairs in feasible_pairs_by_city.values()
                for tower_index, _ in feasible_pairs
            }
        )

        if len(used_tower_indices) == len(towers):
            return towers.copy().reset_index(drop=True), feasible_pairs_by_city, 0

        pruned_towers = towers.loc[used_tower_indices].copy().reset_index(drop=True)
        index_map = {old_index: new_index for new_index, old_index in enumerate(used_tower_indices)}

        pruned_pairs_by_city: Dict[int, List[Tuple[int, float]]] = {}
        for city_index, feasible_pairs in feasible_pairs_by_city.items():
            pruned_pairs_by_city[int(city_index)] = [
                (index_map[int(tower_index)], float(distance))
                for tower_index, distance in feasible_pairs
                if int(tower_index) in index_map
            ]

        return pruned_towers, pruned_pairs_by_city, len(towers) - len(pruned_towers)


class CoverageBuilder:
    def build_for_pair(
        self,
        feasible_pairs_by_city: Dict[int, List[Tuple[int, float]]],
        radius_pair: Tuple[float, float],
    ) -> Dict[int, List[Tuple[int, float]]]:
        radius_small, radius_large = sorted(float(radius) for radius in radius_pair)
        covered_by_city: Dict[int, List[Tuple[int, float]]] = {
            int(city_index): [] for city_index in feasible_pairs_by_city
        }

        for city_index, feasible_pairs in feasible_pairs_by_city.items():
            covered = covered_by_city[int(city_index)]
            for tower_index, distance in feasible_pairs:
                if distance <= radius_small + 1e-9:
                    covered.append((int(tower_index), radius_small))
                    covered.append((int(tower_index), radius_large))
                elif distance <= radius_large + 1e-9:
                    covered.append((int(tower_index), radius_large))

        return covered_by_city


class WarmStartMapper:
    def __init__(self, towers: pd.DataFrame) -> None:
        self.towers = towers.reset_index(drop=True)
        if self.towers.empty:
            self.tree: Optional[BallTree] = None
        else:
            coordinates_rad = np.radians(
                self.towers[["latitude", "longitude"]].to_numpy(dtype=float)
            )
            self.tree = BallTree(coordinates_rad, metric="haversine")

    def nearest_index(self, latitude: float, longitude: float) -> Optional[int]:
        if self.tree is None:
            return None
        _, indices = self.tree.query(
            [[math.radians(float(latitude)), math.radians(float(longitude))]],
            k=1,
        )
        return int(indices[0][0])


class RadiusMatcher:
    def nearest_radius(self, target_radius: float, radii: Iterable[float]) -> float:
        radius_list = [float(radius) for radius in radii]
        return min(radius_list, key=lambda radius: abs(radius - float(target_radius)))
