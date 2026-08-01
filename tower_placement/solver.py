from __future__ import annotations

import gc
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from .config import SolverSettings
from .results import SolveResult
from .candidate_geometry import RadiusMatcher, WarmStartMapper


class GurobiTowerSolver:
    def __init__(self, settings: SolverSettings) -> None:
        self.settings = settings
        self.radius_matcher = RadiusMatcher()

    def status_name(self, status_code: int) -> str:
        status_names = {
            GRB.LOADED: "LOADED",
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.INPROGRESS: "INPROGRESS",
            GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
            GRB.WORK_LIMIT: "WORK_LIMIT",
            GRB.MEM_LIMIT: "MEM_LIMIT",
        }
        return status_names.get(status_code, f"UNKNOWN_STATUS_{status_code}")

    def set_model_parameters(
        self,
        model: gp.Model,
        time_limit: Optional[float],
        mip_gap: Optional[float],
    ) -> None:
        model.Params.OutputFlag = 1 if self.settings.verbose else 0
        model.Params.MIPFocus = self.settings.mip_focus
        if self.settings.threads is not None:
            model.Params.Threads = self.settings.threads
        if time_limit is not None:
            model.Params.TimeLimit = float(time_limit)
        if mip_gap is not None:
            model.Params.MIPGap = float(mip_gap)

    def apply_warm_start(
        self,
        x: gp.tupledict,
        radii: Sequence[float],
        warm_start_towers: pd.DataFrame,
        warm_start_mapper: WarmStartMapper,
    ) -> None:
        if warm_start_towers is None or warm_start_towers.empty:
            return

        for _, var in x.items():
            var.Start = 0.0

        chosen_radius_by_grid_point: Dict[int, float] = {}
        for _, row in warm_start_towers.iterrows():
            best_j = warm_start_mapper.nearest_index(
                float(row["latitude"]),
                float(row["longitude"]),
            )
            if best_j is None:
                continue

            target_radius = float(row["radius"])
            chosen_radius = (
                target_radius
                if target_radius in radii
                else self.radius_matcher.nearest_radius(target_radius, radii)
            )

            existing_radius = chosen_radius_by_grid_point.get(best_j)
            if existing_radius is None or chosen_radius > existing_radius:
                chosen_radius_by_grid_point[best_j] = chosen_radius

        for tower_index, radius in chosen_radius_by_grid_point.items():
            x[tower_index, radius].Start = 1.0

    def solve(
        self,
        cities: pd.DataFrame,
        towers: pd.DataFrame,
        covered_by_city: Dict[int, List[Tuple[int, float]]],
        tower_costs: Dict[float, float],
        lambda_penalty: float,
        model_name: str,
        enforce_both_types: bool,
        time_limit: Optional[float],
        mip_gap: Optional[float],
        warm_start_towers: Optional[pd.DataFrame] = None,
        warm_start_mapper: Optional[WarmStartMapper] = None,
    ) -> SolveResult:
        city_indices = list(cities.index)
        tower_indices = list(towers.index)
        radii = sorted(float(radius) for radius in tower_costs.keys())

        model = gp.Model(model_name)
        try:
            self.set_model_parameters(model, time_limit=time_limit, mip_gap=mip_gap)

            x = model.addVars(tower_indices, radii, vtype=GRB.BINARY, name="x")
            y = model.addVars(city_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

            install_cost_expr = gp.quicksum(
                tower_costs[float(radius)] * x[tower_index, radius]
                for tower_index in tower_indices
                for radius in radii
            )
            interference_penalty_expr = lambda_penalty * gp.quicksum(
                y[city_index] for city_index in city_indices
            )
            model.setObjective(install_cost_expr + interference_penalty_expr, GRB.MINIMIZE)

            for city_index in city_indices:
                city_coverage = covered_by_city.get(int(city_index), [])
                if not city_coverage:
                    return SolveResult(
                        status="INFEASIBLE_COVERAGE",
                        selected_towers=pd.DataFrame(
                            columns=["tower_id", "latitude", "longitude", "radius"]
                        ),
                        objective_value=None,
                        installation_cost=None,
                        interference_penalty=None,
                        num_selected_towers=0,
                        mip_gap=None,
                    )

                coverage_expr = gp.quicksum(
                    x[tower_index, radius] for tower_index, radius in city_coverage
                )
                model.addConstr(coverage_expr >= 1, name=f"coverage_{city_index}")
                model.addConstr(
                    y[city_index] >= coverage_expr - 1,
                    name=f"excess_coverage_{city_index}",
                )
                model.addConstr(y[city_index] >= 0, name=f"nonnegative_y_{city_index}")

            for tower_index in tower_indices:
                model.addConstr(
                    gp.quicksum(x[tower_index, radius] for radius in radii) <= 1,
                    name=f"one_tower_type_{tower_index}",
                )

            if enforce_both_types:
                for radius in radii:
                    model.addConstr(
                        gp.quicksum(x[tower_index, radius] for tower_index in tower_indices) >= 1,
                        name=f"use_radius_{str(radius).replace('.', '_')}",
                    )

            if warm_start_towers is not None and warm_start_mapper is not None:
                self.apply_warm_start(
                    x=x,
                    radii=radii,
                    warm_start_towers=warm_start_towers,
                    warm_start_mapper=warm_start_mapper,
                )

            model.optimize()

            status_name = self.status_name(model.Status)
            mip_gap_value = float(model.MIPGap) if model.SolCount > 0 else None

            if model.SolCount == 0:
                return SolveResult(
                    status=status_name,
                    selected_towers=pd.DataFrame(
                        columns=["tower_id", "latitude", "longitude", "radius"]
                    ),
                    objective_value=None,
                    installation_cost=None,
                    interference_penalty=None,
                    num_selected_towers=0,
                    mip_gap=mip_gap_value,
                )

            selected_rows = []
            for tower_index in tower_indices:
                for radius in radii:
                    if x[tower_index, radius].X > 0.5:
                        selected_rows.append(
                            {
                                "tower_id": towers.loc[tower_index, "tower_id"],
                                "latitude": towers.loc[tower_index, "latitude"],
                                "longitude": towers.loc[tower_index, "longitude"],
                                "radius": float(radius),
                            }
                        )

            selected_towers = pd.DataFrame(selected_rows)
            installation_cost = sum(
                tower_costs[float(row["radius"])] for _, row in selected_towers.iterrows()
            )
            interference_penalty = lambda_penalty * sum(
                y[city_index].X for city_index in city_indices
            )

            return SolveResult(
                status=status_name,
                selected_towers=selected_towers,
                objective_value=float(model.ObjVal),
                installation_cost=float(installation_cost),
                interference_penalty=float(interference_penalty),
                num_selected_towers=int(len(selected_towers)),
                mip_gap=mip_gap_value,
            )
        finally:
            try:
                model.dispose()
            except Exception:
                pass
            gc.collect()
