# Tower Placement Project

This folder is organized around the final modular project state.

## Active Structure

- `main.py` - main orchestration entrypoint for the radius-search workflow.
- `tower_placement/` - reusable solver, search, spatial, cost, and data-loading modules.
- `scripts/` - optional helper scripts for explanation figures.
- `data/` - necessary input data: city files and cost data.
- `results/` - only the final visible outputs: two selected-tower CSVs and two maps.
- `validation/` - validation tools, checker inputs, validation outputs, and run summaries.
- `trash/` - archived experiments, old monolithic scripts, benchmark outputs, presentations, and older generated artifacts.

## Main Results

- `results/selected_towers_germany.csv`
- `results/selected_towers_france.csv`
- `results/germany_tower_coverage_map.png`
- `results/france_tower_coverage_map.png`

## Run The Solver

```powershell
python main.py --instance france
python main.py --instance germany
```

By default, new search runs write to `results/search_runs/<instance>/` so the final result CSVs and maps are not overwritten accidentally.

## Scripts vs Package

`tower_placement/` is the importable project code used by `main.py`.
`scripts/` contains optional one-off utilities around the project, such as plotting or explanation figures.
