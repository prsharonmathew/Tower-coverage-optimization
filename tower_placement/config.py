from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class SolverSettings:
    lambda_penalty: float = 2.0
    n_clusters: int = 4
    phase1_grid_step: float = 0.02
    phase2_grid_step: float = 0.02
    grid_margin: float = 0.1
    phase1_time_limit: Optional[float] = 300.0
    phase2_time_limit: Optional[float] = 300.0
    phase2_mip_gap: Optional[float] = 0.001
    phase1_mip_gap: Optional[float] = None
    kmeans_random_state: int = 42
    verbose: bool = False
    quiet: bool = True
    mip_focus: int = 1
    threads: Optional[int] = None
    earth_radius_km: float = 6371.0
    max_search_radius_km: float = 100.0
    instance_name: str = "germany"


@dataclass(slots=True)
class RadiusSearchSettings:
    input_file: Path
    output_dir: Path
    log_file: Path
    results_file: Path
    best_towers_file: Path
    phase1_num_points: int = 500
    top_k_phase2: int = 3
    phase2_rectangle_radius: int = 5
    top_k_final: int = 1
    phase3_delta: float = 0.9
    phase3_step: float = 0.1
