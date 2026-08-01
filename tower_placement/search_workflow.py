from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import math
import pandas as pd

from .config import RadiusSearchSettings
from .costs import RadiusCostService
from .logging_utils import SearchLogger
from .pipeline import TwoPhaseRadiusEvaluator
from .radius_pairs import RadiusPairGenerator
from .results import PhaseContext, RadiusPairEvaluation


class RadiusSearchWorkflow:
    def __init__(
        self,
        evaluator: TwoPhaseRadiusEvaluator,
        search_settings: RadiusSearchSettings,
        logger: SearchLogger,
    ) -> None:
        self.evaluator = evaluator
        self.search_settings = search_settings
        self.logger = logger
        self.cost_service = RadiusCostService()
        self.pair_generator = RadiusPairGenerator(search_settings, self.cost_service)
        self.records: List[dict] = []
        self.evaluated_pairs: Set[Tuple[float, float]] = set()
        self.resume_cache: Dict[Tuple[float, float], RadiusPairEvaluation] = {}

    def load_resume(self) -> None:
        results_file = self.search_settings.results_file
        if not results_file.exists():
            return

        resume_df = pd.read_csv(results_file)
        if resume_df.empty:
            return

        loaded = 0
        for row in resume_df.to_dict(orient="records"):
            key = self.cost_service.normalize_pair((row["r1"], row["r2"]))
            if key in self.resume_cache:
                continue

            objective_value = self._clean_optional_float(row.get("objective_value"))
            installation_cost = self._clean_optional_float(row.get("installation_cost"))
            interference_penalty = self._clean_optional_float(row.get("interference_penalty"))
            mip_gap = self._clean_optional_float(row.get("mip_gap"))
            runtime_seconds = self._clean_optional_float(row.get("runtime_seconds"))
            num_selected_towers = int(row.get("num_selected_towers", 0) or 0)
            solver_status = str(row.get("solver_status", "UNKNOWN"))

            evaluation = RadiusPairEvaluation(
                radius_pair=key,
                selected_towers=pd.DataFrame(
                    columns=["tower_id", "latitude", "longitude", "radius"]
                ),
                objective_value=objective_value,
                installation_cost=installation_cost,
                interference_penalty=interference_penalty,
                num_selected_towers=num_selected_towers,
                solver_status=solver_status,
                mip_gap=mip_gap,
                runtime_seconds=runtime_seconds,
            )
            self.resume_cache[key] = evaluation
            self.evaluated_pairs.add(key)
            self.records.append(
                {
                    "r1": key[0],
                    "r2": key[1],
                    "objective_value": objective_value,
                    "installation_cost": installation_cost,
                    "interference_penalty": interference_penalty,
                    "num_selected_towers": num_selected_towers,
                    "solver_status": solver_status,
                    "mip_gap": mip_gap,
                    "runtime_seconds": runtime_seconds,
                    "c_r1": self.cost_service.radius_cost(key[0]),
                    "c_r2": self.cost_service.radius_cost(key[1]),
                }
            )
            loaded += 1

        if loaded > 0:
            self.logger.info(
                f"Resume loaded: {loaded} previously evaluated radius pairs from {results_file}"
            )

    def evaluate_pair(
        self,
        phase1_contexts: Sequence[PhaseContext],
        phase2_context: PhaseContext,
        radius_pair: Tuple[float, float],
    ) -> RadiusPairEvaluation:
        normalized_pair = self.cost_service.normalize_pair(radius_pair)
        if normalized_pair in self.resume_cache:
            return self.resume_cache[normalized_pair]
        if normalized_pair in self.evaluated_pairs:
            return RadiusPairEvaluation(
                radius_pair=normalized_pair,
                selected_towers=pd.DataFrame(
                    columns=["tower_id", "latitude", "longitude", "radius"]
                ),
                objective_value=None,
                installation_cost=None,
                interference_penalty=None,
                num_selected_towers=0,
                solver_status="SKIPPED_DUPLICATE",
                mip_gap=None,
                runtime_seconds=None,
            )

        self.evaluated_pairs.add(normalized_pair)

        try:
            evaluation = self.evaluator.evaluate_radius_pair(
                phase1_contexts=phase1_contexts,
                phase2_context=phase2_context,
                radius_pair=normalized_pair,
            )
        except Exception as exc:
            evaluation = RadiusPairEvaluation(
                radius_pair=normalized_pair,
                selected_towers=pd.DataFrame(
                    columns=["tower_id", "latitude", "longitude", "radius"]
                ),
                objective_value=None,
                installation_cost=None,
                interference_penalty=None,
                num_selected_towers=0,
                solver_status=f"ERROR: {type(exc).__name__}",
                mip_gap=None,
                runtime_seconds=None,
            )

        record = evaluation.to_record()
        record["c_r1"] = self.cost_service.radius_cost(normalized_pair[0])
        record["c_r2"] = self.cost_service.radius_cost(normalized_pair[1])
        self.records.append(record)
        self._save_results_snapshot()

        if evaluation.objective_value is not None:
            self.logger.info(
                f"Pair={self.cost_service.pair_label(normalized_pair)} | "
                f"Obj={evaluation.objective_value:.4f} | "
                f"Install={evaluation.installation_cost:.4f} | "
                f"Penalty={evaluation.interference_penalty:.4f} | "
                f"Towers={evaluation.num_selected_towers} | "
                f"Gap={evaluation.mip_gap} | "
                f"Time={evaluation.runtime_seconds:.2f}s | "
                f"Status={evaluation.solver_status}"
            )
        else:
            self.logger.info(
                f"Pair={self.cost_service.pair_label(normalized_pair)} | "
                f"FAILED | Status={evaluation.solver_status}"
            )
        return evaluation

    def run(
        self,
        cities: pd.DataFrame,
        phase1_contexts: Sequence[PhaseContext],
        phase2_context: PhaseContext,
    ) -> Dict[str, object]:
        self.load_resume()
        self.logger.info("=" * 80)
        self.logger.info("PHASE 1 : COARSE SEARCH")
        self.logger.info("=" * 80)

        phase1_results: List[RadiusPairEvaluation] = []
        for pair in self.pair_generator.generate_phase1_pairs():
            result = self.evaluate_pair(phase1_contexts, phase2_context, pair)
            if result.objective_value is not None:
                phase1_results.append(result)

        phase1_results.sort(key=lambda item: item.objective_value)
        top_phase1 = phase1_results[: self.search_settings.top_k_phase2]
        self.logger.info(
            f"Top coarse pairs: {self.pair_generator.format_pairs([item.radius_pair for item in top_phase1])}"
        )

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 2 : INTEGER LOCAL RECTANGLE SEARCH")
        self.logger.info("=" * 80)

        phase2_results: List[RadiusPairEvaluation] = []
        for result in top_phase1:
            self.logger.info(
                f"Searching neighborhood around {self.cost_service.pair_label(result.radius_pair)}"
            )
            for pair in self.pair_generator.generate_phase2_pairs(result.radius_pair):
                pair_result = self.evaluate_pair(phase1_contexts, phase2_context, pair)
                if pair_result.objective_value is not None:
                    phase2_results.append(pair_result)

        phase2_results.sort(key=lambda item: item.objective_value)
        top_phase2 = phase2_results[: self.search_settings.top_k_final]
        self.logger.info(
            f"Top refined pairs: {self.pair_generator.format_pairs([item.radius_pair for item in top_phase2])}"
        )

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 3 : DECIMAL LOCAL REFINEMENT")
        self.logger.info("=" * 80)

        phase3_results: List[RadiusPairEvaluation] = []
        for result in top_phase2:
            self.logger.info(
                f"Refining around {self.cost_service.pair_label(result.radius_pair)}"
            )
            for pair in self.pair_generator.generate_phase3_pairs(result.radius_pair):
                pair_result = self.evaluate_pair(phase1_contexts, phase2_context, pair)
                if pair_result.objective_value is not None:
                    phase3_results.append(pair_result)

        valid_results = [
            record for record in (phase1_results + phase2_results + phase3_results)
            if record.objective_value is not None
        ]
        valid_results.sort(key=lambda item: item.objective_value)

        if not valid_results:
            raise RuntimeError("No feasible radius pair result was found.")

        search_best_result = valid_results[0]

        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINAL BEST-PAIR REOPTIMIZATION")
        self.logger.info("=" * 80)
        self.logger.info(
            "Rerunning the full two-phase optimization for the winning radius pair "
            f"{self.cost_service.pair_label(search_best_result.radius_pair)} "
            "to get the final best cost breakdown."
        )

        best_result = self._rerun_best_pair(
            phase1_contexts=phase1_contexts,
            phase2_context=phase2_context,
            radius_pair=search_best_result.radius_pair,
        )

        self._save_best_towers(best_result)
        self._save_results_snapshot()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINAL BEST RESULT")
        self.logger.info("=" * 80)
        self.logger.info(f"Best pair: {self.cost_service.pair_label(best_result.radius_pair)}")
        self.logger.info(f"Objective value: {best_result.objective_value}")
        self.logger.info(f"Installation cost: {best_result.installation_cost}")
        self.logger.info(f"Interference penalty: {best_result.interference_penalty}")
        self.logger.info(f"Selected towers: {best_result.num_selected_towers}")
        self.logger.info(f"Solver status: {best_result.solver_status}")
        self.logger.info(f"MIP gap: {best_result.mip_gap}")
        self.logger.info(f"Saved results: {self.search_settings.results_file}")
        self.logger.info(f"Saved best towers: {self.search_settings.best_towers_file}")

        return {
            "search_best_result": search_best_result,
            "best_result": best_result,
            "all_records": self.records,
        }

    @staticmethod
    def _clean_optional_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric

    def _rerun_best_pair(
        self,
        phase1_contexts: Sequence[PhaseContext],
        phase2_context: PhaseContext,
        radius_pair: Tuple[float, float],
    ) -> RadiusPairEvaluation:
        final_result = self.evaluator.evaluate_radius_pair(
            phase1_contexts=phase1_contexts,
            phase2_context=phase2_context,
            radius_pair=radius_pair,
        )

        if final_result.objective_value is not None:
            self.logger.info(
                f"Final rerun for {self.cost_service.pair_label(final_result.radius_pair)} | "
                f"Objective={final_result.objective_value:.4f} | "
                f"Installation={final_result.installation_cost:.4f} | "
                f"Interference={final_result.interference_penalty:.4f} | "
                f"Towers={final_result.num_selected_towers} | "
                f"Gap={final_result.mip_gap} | "
                f"Status={final_result.solver_status}"
            )
        else:
            self.logger.info(
                f"Final rerun for {self.cost_service.pair_label(final_result.radius_pair)} failed | "
                f"Status={final_result.solver_status}"
            )

        return final_result

    def _save_results_snapshot(self) -> None:
        if not self.records:
            return
        output_df = pd.DataFrame(self.records).sort_values(
            by=["objective_value", "r1", "r2"],
            na_position="last",
        )
        output_df.to_csv(self.search_settings.results_file, index=False)

    def _save_best_towers(self, best_result: RadiusPairEvaluation) -> None:
        if best_result.selected_towers.empty:
            return
        output_df = best_result.selected_towers[["latitude", "longitude", "radius"]].copy()
        output_df.to_csv(self.search_settings.best_towers_file, index=False)
