from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.cluster import KMeans

from .config import SolverSettings
from .costs import RadiusCostService
from .logging_utils import SearchLogger
from .results import PhaseContext, PhaseOneResult, RadiusPairEvaluation
from .solver import GurobiTowerSolver
from .candidate_geometry import (
    CandidateTowerGenerator,
    CoverageBuilder,
    FeasiblePairBuilder,
    TowerPruner,
    WarmStartMapper,
)


class PhaseContextBuilder:
    def __init__(self, settings: SolverSettings, logger: SearchLogger) -> None:
        self.settings = settings
        self.logger = logger
        self.grid_generator = CandidateTowerGenerator()
        self.feasible_pair_builder = FeasiblePairBuilder(settings.earth_radius_km)
        self.pruner = TowerPruner()

    def build_phase1_contexts(self, cities: pd.DataFrame) -> Tuple[pd.DataFrame, List[PhaseContext]]:
        city_array = cities[["latitude", "longitude"]].to_numpy()
        kmeans = KMeans(
            n_clusters=self.settings.n_clusters,
            n_init=10,
            random_state=self.settings.kmeans_random_state,
        )
        cities_with_zone = cities.copy()
        cities_with_zone["zone"] = kmeans.fit_predict(city_array)

        contexts: List[PhaseContext] = []
        for cluster_id in sorted(cities_with_zone["zone"].unique()):
            cluster_cities = cities_with_zone[cities_with_zone["zone"] == cluster_id].copy()
            towers = self.grid_generator.generate(
                cluster_cities,
                grid_step=self.settings.phase1_grid_step,
                grid_margin=self.settings.grid_margin,
            )
            feasible_pairs_by_city = self.feasible_pair_builder.build(
                cluster_cities,
                towers,
                max_radius_km=self.settings.max_search_radius_km,
            )
            towers, feasible_pairs_by_city, pruned_count = self.pruner.prune(
                towers,
                feasible_pairs_by_city,
            )
            if pruned_count > 0:
                self.logger.debug(
                    f"Cluster {cluster_id}: pruned {pruned_count} useless tower candidates."
                )

            contexts.append(
                PhaseContext(
                    cluster_id=int(cluster_id),
                    cities=cluster_cities,
                    towers=towers,
                    feasible_pairs_by_city=feasible_pairs_by_city,
                )
            )
        return cities_with_zone, contexts

    def build_phase2_context(self, cities: pd.DataFrame) -> PhaseContext:
        towers = self._load_phase2_towers(cities)
        feasible_pairs_by_city = self.feasible_pair_builder.build(
            cities,
            towers,
            max_radius_km=self.settings.max_search_radius_km,
        )
        towers, feasible_pairs_by_city, pruned_count = self.pruner.prune(
            towers,
            feasible_pairs_by_city,
        )
        if pruned_count > 0:
            self.logger.debug(f"Phase 2: pruned {pruned_count} useless tower candidates.")

        return PhaseContext(
            cluster_id=None,
            cities=cities,
            towers=towers,
            feasible_pairs_by_city=feasible_pairs_by_city,
            warm_start_mapper=WarmStartMapper(towers),
        )

    def _load_phase2_towers(self, cities: pd.DataFrame) -> pd.DataFrame:
        return self.grid_generator.generate(
            cities,
            grid_step=self.settings.phase2_grid_step,
            grid_margin=self.settings.grid_margin,
        )


class TwoPhaseRadiusEvaluator:
    def __init__(self, settings: SolverSettings, logger: SearchLogger) -> None:
        self.settings = settings
        self.logger = logger
        self.cost_service = RadiusCostService()
        self.coverage_builder = CoverageBuilder()
        self.solver = GurobiTowerSolver(settings)
        self.context_builder = PhaseContextBuilder(settings, logger)

    def precompute_contexts(self, cities: pd.DataFrame) -> Tuple[pd.DataFrame, List[PhaseContext], PhaseContext]:
        self.logger.debug("\nPrecomputing reusable phase-1 and phase-2 contexts...")
        cities_with_zone, phase1_contexts = self.context_builder.build_phase1_contexts(cities)
        phase2_context = self.context_builder.build_phase2_context(cities_with_zone)
        return cities_with_zone, phase1_contexts, phase2_context

    def _coverage_for_pair(
        self,
        context: PhaseContext,
        radius_pair: Tuple[float, float],
    ) -> Dict[int, List[Tuple[int, float]]]:
        normalized_pair = self.cost_service.normalize_pair(radius_pair)
        return self.coverage_builder.build_for_pair(
            context.feasible_pairs_by_city,
            normalized_pair,
        )

    def run_phase1(
        self,
        phase1_contexts: Sequence[PhaseContext],
        tower_costs: Dict[float, float],
        radius_pair: Tuple[float, float],
    ) -> PhaseOneResult:
        all_selected_towers = []
        total_objective_value = 0.0
        total_installation_cost = 0.0
        total_interference_penalty = 0.0
        total_num_selected_towers = 0
        status_parts: List[str] = []
        had_any_solution = False

        for context in phase1_contexts:
            covered_by_city = self._coverage_for_pair(context, radius_pair)
            result = self.solver.solve(
                cities=context.cities,
                towers=context.towers,
                covered_by_city=covered_by_city,
                tower_costs=tower_costs,
                lambda_penalty=self.settings.lambda_penalty,
                model_name=f"tower_placement_cluster_{context.cluster_id}_{self.cost_service.pair_label(radius_pair)}",
                enforce_both_types=False,
                time_limit=self.settings.phase1_time_limit,
                mip_gap=self.settings.phase1_mip_gap,
            )

            status_parts.append(f"cluster_{context.cluster_id}:{result.status}")
            if not result.selected_towers.empty:
                cluster_selected = result.selected_towers.copy()
                cluster_selected["cluster"] = context.cluster_id
                all_selected_towers.append(cluster_selected)

            if result.objective_value is not None:
                had_any_solution = True
                total_objective_value += float(result.objective_value)
                total_installation_cost += float(result.installation_cost)
                total_interference_penalty += float(result.interference_penalty)
                total_num_selected_towers += int(result.num_selected_towers)

        if all_selected_towers:
            combined_selected_towers = pd.concat(all_selected_towers, ignore_index=True)
            combined_selected_towers = combined_selected_towers.drop_duplicates(
                subset=["latitude", "longitude", "radius"]
            ).reset_index(drop=True)
        else:
            combined_selected_towers = pd.DataFrame(
                columns=["tower_id", "latitude", "longitude", "radius", "cluster"]
            )

        return PhaseOneResult(
            status="SUCCESS" if had_any_solution else "NO_PHASE1_SOLUTION",
            status_summary="; ".join(status_parts),
            selected_towers=combined_selected_towers,
            objective_value=total_objective_value if had_any_solution else None,
            installation_cost=total_installation_cost if had_any_solution else None,
            interference_penalty=total_interference_penalty if had_any_solution else None,
            num_selected_towers=total_num_selected_towers,
        )

    def run_phase2(
        self,
        phase2_context: PhaseContext,
        phase1_selected_towers: pd.DataFrame,
        tower_costs: Dict[float, float],
        radius_pair: Tuple[float, float],
    ):
        covered_by_city = self._coverage_for_pair(phase2_context, radius_pair)
        return self.solver.solve(
            cities=phase2_context.cities,
            towers=phase2_context.towers,
            covered_by_city=covered_by_city,
            tower_costs=tower_costs,
            lambda_penalty=self.settings.lambda_penalty,
            model_name=f"phase2_global_{self.cost_service.pair_label(radius_pair)}",
            enforce_both_types=True,
            time_limit=self.settings.phase2_time_limit,
            mip_gap=self.settings.phase2_mip_gap,
            warm_start_towers=phase1_selected_towers,
            warm_start_mapper=phase2_context.warm_start_mapper,
        )

    def evaluate_radius_pair(
        self,
        phase1_contexts: Sequence[PhaseContext],
        phase2_context: PhaseContext,
        radius_pair: Tuple[float, float],
    ) -> RadiusPairEvaluation:
        start_time = time.time()
        normalized_pair = self.cost_service.normalize_pair(radius_pair)
        tower_costs = self.cost_service.tower_costs_for_pair(normalized_pair)

        phase1_result = self.run_phase1(phase1_contexts, tower_costs, normalized_pair)
        phase2_result = self.run_phase2(
            phase2_context,
            phase1_result.selected_towers,
            tower_costs,
            normalized_pair,
        )

        return RadiusPairEvaluation(
            radius_pair=normalized_pair,
            selected_towers=phase2_result.selected_towers,
            objective_value=phase2_result.objective_value,
            installation_cost=phase2_result.installation_cost,
            interference_penalty=phase2_result.interference_penalty,
            num_selected_towers=phase2_result.num_selected_towers,
            solver_status=phase2_result.status,
            mip_gap=phase2_result.mip_gap,
            runtime_seconds=time.time() - start_time,
        )
