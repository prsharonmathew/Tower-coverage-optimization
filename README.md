# Tower Coverage Optimization

Find the best way in which 6G communication towers can be placed in the selected
country or region, while ensuring complete coverage and minimum inference.

## Active Structure

- `main.py` - main entrypoint of the entire workflow.
- `tower_placement/` - reusable solver, search, spatial, cost and data-loading modules.
- `scripts/` - optional helper scripts for explanation figures.
- `data/` - necessary input data: city files and cost data.
- `results/` - only the final visible outputs: two selected-tower CSVs and two maps.
- `validation/` - validation tools, checker inputs, validation outputs, and run summaries.

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

## Requirements

This project requires Python 3.12 or later and the following Python libraries:

- `gurobipy` - MILP optimization with the Gurobi solver.
- `pandas` - data loading, cleaning, and CSV output
- `numpy` - numerical array operations
- `scikit-learn` - clustering and spatial search using `KMeans` and `BallTree`
- `matplotlib` - plotting helper figures

Install the required libraries with:

```bash
pip install pandas numpy scikit-learn gurobipy matplotlib


