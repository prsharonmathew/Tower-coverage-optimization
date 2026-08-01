from __future__ import annotations

from typing import Dict, Tuple


class RadiusCostService:
    def radius_cost(self, radius_km: float) -> float:
        r = float(radius_km)
        if 5 <= r <= 20:
            return (
                0.00003898883009994121 * r**3
                - 0.0005848324514991181 * r**2
                + 0.000818342151675485 * r
                + 0.9056554967666078
            )
        if 20 < r <= 35:
            return (
                -0.00004679600235155791 * r**3
                + 0.004562257495590829 * r**2
                - 0.10212345679012345 * r
                + 1.591934156378601
            )
        if 35 < r <= 50:
            return (
                0.00005930629041740153 * r**3
                - 0.006578483245149912 * r**2
                + 0.2878024691358025 * r
                - 2.957201646090535
            )
        if 50 < r <= 100:
            return (
                -0.00001544973544973545 * r**3
                + 0.004634920634920635 * r**2
                - 0.27286772486772487 * r
                + 6.387301587301588
            )
        raise ValueError("Radius must be in [5, 100].")

    @staticmethod
    def normalize_pair(radius_pair: Tuple[float, float], ndigits: int = 3) -> Tuple[float, float]:
        r1 = round(float(radius_pair[0]), ndigits)
        r2 = round(float(radius_pair[1]), ndigits)
        ordered = tuple(sorted((r1, r2)))
        return ordered

    def pair_label(self, radius_pair: Tuple[float, float]) -> str:
        r1, r2 = self.normalize_pair(radius_pair)
        return f"({r1:.1f}, {r2:.1f})"

    def tower_costs_for_pair(self, radius_pair: Tuple[float, float]) -> Dict[float, float]:
        r1, r2 = self.normalize_pair(radius_pair)
        if r1 == r2:
            raise ValueError("Radius pair must contain two distinct radii.")
        return {
            float(r1): self.radius_cost(r1),
            float(r2): self.radius_cost(r2),
        }
