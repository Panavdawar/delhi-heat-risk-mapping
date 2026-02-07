# Delhi Heat Risk Mapping

Data-to-decision workflow for assessing heat-risk in Delhi (weather, derived indices, and spatial joins) with reproducible day-by-day notebooks and lightweight visualization.

## What’s inside
- **Notebooks (`notebooks/day1...day5`)**: Download → clean/features → baseline ML → risk labeling → mapping.
- **Source (`src/`)**: helpers for loading, preprocessing, risk modeling, and Streamlit app prototype.
- **Outputs (`outputs/`)**: cached maps/plots for quick review (models are ignored to avoid huge binaries).
- **Data (`data/`)**: processed CSV/GeoJSON artifacts; raw NetCDF/CSV expected but not committed.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run the notebooks
Open Jupyter and step through the pipeline (order matters):
1. `notebooks/day1_data_download.ipynb` – pulls raw weather inputs.
2. `notebooks/day2_cleaning_features.ipynb` – cleans + engineers features.
3. `notebooks/day3_baseline_ml.ipynb` – trains RF baseline, feature importance, multivariate analysis helpers.
4. `notebooks/day4_risk_labeling.ipynb` – labels risk scores.
5. `notebooks/day5_mapping.ipynb` – joins to spatial layers and renders maps.

### Command-line example
```bash
python -m src.main
```
Loads processed NetCDF/CSV, applies risk model, performs spatial join, and writes `dwarka_heat_risk.html` to `outputs/maps/`.

### Streamlit prototype
```bash
streamlit run src/streamlit_app.py
```

## Key pieces
- `src/_risk_metrics.py`: RF baseline trainer, improved feature-importance visuals, multivariate diagnostics, and feature–target correlation helper.
- `src/spatial_mapping.py` & `geo/`: utilities and shapes for Delhi districts/grid overlays.
- `outputs/maps/`: ready-made HTML heatmaps for quick inspection.

## Data expectations
- Weather grid NetCDF files under `data/preprocessed/raw/` (filename pattern in `src/main.py`).
- GeoJSON boundaries in `geo/` (e.g., `delhi_districts.geojson`, grid variants).
- Processed intermediate CSVs are kept to shorten reruns.

## Contributing / housekeeping
- Large binaries (models) are excluded via `.gitignore`. Please keep generated models/artifacts out of git or use external storage/LFS.
- Use descriptive commit messages; branch from `main` and open PRs as usual.

## License
See `LICENSE`.
