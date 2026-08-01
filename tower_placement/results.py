from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(slots=True)
class PhaseContext:
    cluster_id: Optional[int]
    cities: pd.DataFrame
    towers: pd.DataFrame
    feasible_pairs_by_city: Dict[int, List[Tuple[int, float]]]
    warm_start_mapper: Optional["WarmStartMapper"] = None


@dataclass(slots=True)
class SolveResult:
    status: str
    selected_towers: pd.DataFrame
    objective_value: Optional[float]
    installation_cost: Optional[float]
    interference_penalty: Optional[float]
    num_selected_towers: int
    mip_gap: Optional[float]


@dataclass(slots=True)
class PhaseOneResult:
    status: str
    status_summary: str
    selected_towers: pd.DataFrame
    objective_value: Optional[float]
    installation_cost: Optional[float]
    interference_penalty: Optional[float]
    num_selected_towers: int


@dataclass(slots=True)
class RadiusPairEvaluation:
    radius_pair: Tuple[float, float]
    selected_towers: pd.DataFrame
    objective_value: Optional[float]
    installation_cost: Optional[float]
    interference_penalty: Optional[float]
    num_selected_towers: int
    solver_status: str
    mip_gap: Optional[float]
    runtime_seconds: Optional[float]

    def to_record(self) -> dict:
        return {
            "r1": self.radius_pair[0],
            "r2": self.radius_pair[1],
            "objective_value": self.objective_value,
            "installation_cost": self.installation_cost,
            "interference_penalty": self.interference_penalty,
            "num_selected_towers": self.num_selected_towers,
            "solver_status": self.solver_status,
            "mip_gap": self.mip_gap,
            "runtime_seconds": self.runtime_seconds,
        }


# Forward declaration for typing only.
class WarmStartMapper:  # pragma: no cover
    pass
