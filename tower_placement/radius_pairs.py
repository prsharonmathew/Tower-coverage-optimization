from __future__ import annotations

import math
from typing import List, Sequence, Set, Tuple

from .config import RadiusSearchSettings
from .costs import RadiusCostService


class RadiusPairGenerator:
    def __init__(self, settings: RadiusSearchSettings, cost_service: RadiusCostService) -> None:
        self.settings = settings
        self.cost_service = cost_service

    def generate_phase1_pairs(self) -> List[Tuple[float, float]]:
        grid_size = int(math.sqrt(self.settings.phase1_num_points))
        if grid_size < 2:
            raise ValueError("PHASE1_NUM_POINTS must be at least 4.")

        r_values = []
        for index in range(grid_size):
            value = 5 + (95 * index) / (grid_size - 1)
            r_values.append(round(value, 1))

        return self._unique_strict_pairs(
            (float(r1), float(r2))
            for r1 in r_values
            for r2 in r_values
        )

    def generate_phase2_pairs(self, center_pair: Tuple[float, float]) -> List[Tuple[float, float]]:
        center_r1, center_r2 = self.cost_service.normalize_pair(center_pair)
        c1 = int(round(center_r1))
        c2 = int(round(center_r2))

        r1_min = max(5, c1 - self.settings.phase2_rectangle_radius)
        r1_max = min(100, c1 + self.settings.phase2_rectangle_radius)
        r2_min = max(5, c2 - self.settings.phase2_rectangle_radius)
        r2_max = min(100, c2 + self.settings.phase2_rectangle_radius)

        return self._unique_strict_pairs(
            (float(r1), float(r2))
            for r1 in range(r1_min, r1_max + 1)
            for r2 in range(r2_min, r2_max + 1)
        )

    def generate_phase3_pairs(self, center_pair: Tuple[float, float]) -> List[Tuple[float, float]]:
        best_r1, best_r2 = self.cost_service.normalize_pair(center_pair)

        r1_values = []
        current = best_r1 - self.settings.phase3_delta
        while current <= best_r1 + self.settings.phase3_delta + 1e-9:
            r1_values.append(round(current, 1))
            current += self.settings.phase3_step

        r2_values = []
        current = best_r2 - self.settings.phase3_delta
        while current <= best_r2 + self.settings.phase3_delta + 1e-9:
            r2_values.append(round(current, 1))
            current += self.settings.phase3_step

        return self._unique_strict_pairs(
            (float(r1), float(r2))
            for r1 in r1_values
            for r2 in r2_values
        )

    def _unique_strict_pairs(self, pairs) -> List[Tuple[float, float]]:
        unique_pairs: List[Tuple[float, float]] = []
        seen: Set[Tuple[float, float]] = set()
        for pair in pairs:
            normalized = self.cost_service.normalize_pair(pair)
            if normalized[0] >= normalized[1]:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_pairs.append(normalized)
        return unique_pairs

    def format_pairs(self, pairs: Sequence[Tuple[float, float]]) -> str:
        return ", ".join(self.cost_service.pair_label(pair) for pair in pairs)
