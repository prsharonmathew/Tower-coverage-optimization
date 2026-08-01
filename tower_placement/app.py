from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RadiusSearchSettings, SolverSettings
from .data_loader import CityDataLoader
from .logging_utils import SearchLogger
from .pipeline import TwoPhaseRadiusEvaluator
from .search_workflow import RadiusSearchWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True, slots=True)
class InstanceConfig:
    name: str
    city_file: Path


INSTANCE_CONFIGS = {
    "germany": InstanceConfig(
        name="germany",
        city_file=DATA_DIR / "cities" / "cities_de_50k.txt",
    ),
    "france": InstanceConfig(
        name="france",
        city_file=DATA_DIR / "cities" / "cities_fr_30k.txt",
    ),
}


class RadiusSearchProgram:
    def __init__(
        self,
        instance_name: str = "france",
        output_dir: Path | None = None,
    ) -> None:
        if instance_name not in INSTANCE_CONFIGS:
            valid_names = ", ".join(sorted(INSTANCE_CONFIGS))
            raise ValueError(f"Unknown instance '{instance_name}'. Use one of: {valid_names}")

        instance = INSTANCE_CONFIGS[instance_name]
        run_output_dir = output_dir or RESULTS_DIR / "search_runs" / instance.name

        self.solver_settings = SolverSettings(
            lambda_penalty=2.0,
            n_clusters=4,
            phase1_grid_step=0.02,
            phase2_grid_step=0.02,
            grid_margin=0.1,
            phase1_time_limit=300.0,
            phase2_time_limit=300.0,
            phase2_mip_gap=0.001,
            phase1_mip_gap=None,
            kmeans_random_state=42,
            verbose=False,
            quiet=True,
            mip_focus=1,
            threads=None,
            earth_radius_km=6371.0,
            max_search_radius_km=100.0,
            instance_name=instance.name,
        )
        self.search_settings = RadiusSearchSettings(
            input_file=instance.city_file,
            output_dir=run_output_dir,
            log_file=run_output_dir / "progress_log.txt",
            results_file=run_output_dir / "radius_search_results.csv",
            best_towers_file=run_output_dir / "best_selected_towers.csv",
            phase1_num_points=500,
            top_k_phase2=3,
            phase2_rectangle_radius=5,
            top_k_final=1,
            phase3_delta=0.9,
            phase3_step=0.1,
        )
        self.logger = SearchLogger(quiet=True, log_file=self.search_settings.log_file)
        self.loader = CityDataLoader()
        self.evaluator = TwoPhaseRadiusEvaluator(self.solver_settings, self.logger)
        self.workflow = RadiusSearchWorkflow(
            evaluator=self.evaluator,
            search_settings=self.search_settings,
            logger=self.logger,
        )

    def run(self) -> None:
        self.search_settings.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.search_settings.results_file.exists():
            self.logger.clear()
        else:
            self.logger.info(
                f"Resuming existing search from {self.search_settings.results_file}"
            )

        self.logger.debug("Loading city data...")
        cities = self.loader.load(self.search_settings.input_file)
        self.logger.debug(
            f"Loaded {len(cities)} cities from {self.search_settings.input_file}"
        )
        self.logger.debug(
            "Hierarchical 3-phase radius search using Method 2 optimization engine."
        )

        cities_with_zone, phase1_contexts, phase2_context = (
            self.evaluator.precompute_contexts(cities)
        )
        self.workflow.run(cities_with_zone, phase1_contexts, phase2_context)
