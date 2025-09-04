## 20-Hour Water Level Forecasting (Rewaghat Station)

Predict Rewaghat station water level **20 hours ahead** using an XGBoost regression model built from historical gauge data (Chatia & Rewaghat) and aggregated rainfall records.

This repo contains:
- `20_hours_predicting_final.ipynb` – full end‑to‑end workflow (data merge, feature engineering, tuning, training, evaluation, seasonal analyses)
- Excel source data (`Chatia_*.xlsx`, `Rewaghat_*.xlsx`, `rainfall_data.xlsx`)
- `pyproject.toml` – dependency + metadata
- `main.py` – placeholder script entry point (extend for CLI / service usage)

> The notebook trains a model that predicts the Rewaghat water level 20 hours into the future with custom accuracy scoring (±0.15 m tolerance window).

---
### Key Features
- 20‑hour ahead target: `Rewaghat_20h_future`
- Rich time‑series feature set:
	- Lags (1–20) for both Chatia & Rewaghat
	- Rolling stats (mean/std/max) over 3, 6, 12, 24 hours
	- Rainfall aggregates: 6h / 12h / 24h / 48h / 7d sums, 24h max & mean, lag, hours since rain
	- Temporal signals: hour, month, day-of-year + cyclical encodings (sin/cos)
	- Short & medium change deltas (1h, 6h)
- Sample weighting to emphasize recent years
- Randomized hyperparameter search with a **custom accuracy scorer** (fraction of predictions within ±0.15 m)
- Seasonal segmentation: monsoon (Jun–Oct) vs. dry (Nov–May)

### Data Split Strategy
Time‑ordered (no leakage):
- Train: all data except last 2 years
- Validation: year −2 to year −1
- Test: most recent 1 year (plus focused 4‑month monsoon window & dry-season subset)

### Model Artifact (created after running notebook)
`xgb_rewaghat_model_bundle_20hr.pkl` – dictionary with:
```python
{
	'model': XGBRegressor(...),
	'feature_names': [... feature column order ...]
}
```

### Quick Start
```bash
# (Option A) Using uv (uv.lock present)
pip install uv --upgrade
uv sync

# (Option B) Classic venv + pip
python -m venv .venv
source .venv/Scripts/activate  # Git Bash / Windows
pip install -e .
```

Launch Jupyter / VS Code and run the notebook:
```bash
python -m ipykernel install --user --name cwc20
```

### Reproducing Training
1. Ensure Excel data files are in repo root (as currently structured).
2. Open and run all cells in `20_hours_predicting_final.ipynb`.
3. Hyperparameter search persists best params to `best_xgb_params_20hr.pkl`.
4. Final model + feature list saved to `xgb_rewaghat_model_bundle_20hr.pkl`.
5. Evaluation cells output metrics & plots (distribution, time series, seasonal subsets).

### Using the Trained Model Programmatically
```python
import joblib
import pandas as pd
from pathlib import Path

bundle = joblib.load('xgb_rewaghat_model_bundle_20hr.pkl')
model = bundle['model']
feature_names = bundle['feature_names']

# df should be preprocessed with the same feature engineering pipeline
# producing all columns in feature_names in identical order
X = your_feature_engineered_dataframe[feature_names]
pred_20h = model.predict(X)
```

### Extending (`main.py` idea)
You can convert the notebook logic into a modular pipeline (e.g., `src/` package) and expose:
- `prepare_data()`
- `engineer_features(df)`
- `train_model(X_train, y_train, X_val, y_val)`
- `evaluate(model, X, y)`

### Project Structure (current)
```
.
├── 20_hours_predicting_final.ipynb
├── Chatia_train.xlsx / Chatia_test.xlsx
├── Rewaghat_train.xlsx / Rewaghat_test.xlsx
├── rainfall_data.xlsx
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### Recommended Enhancements (Future)
- Add a feature engineering module & unit tests
- Add CLI (e.g., `python -m cwc_forecast predict --input latest.xlsx`)
- Integrate model drift monitoring & retraining schedule
- Package rainfall / station ingestion into reproducible scripts
- Add `.gitignore` (if not already committed) excluding large/sensitive artifacts & virtual env

### Potential .gitignore Template
```gitignore
__pycache__/
.venv/
*.pyc
.ipynb_checkpoints/
*.log
*.pkl
# Uncomment below to keep raw spreadsheets private (optional):
# *.xlsx
```

### Notes on Data & Publication
Ensure the Excel files do not contain confidential or personally identifiable information before pushing publicly. If licensing or usage constraints apply, document them here.

### Python / Dependencies
- Python: 3.13+
- Core libs: pandas, numpy, scikit-learn, xgboost, matplotlib, joblib, openpyxl

### License
Add an explicit license (e.g., MIT, Apache 2.0) depending on organizational policy.

### Citation (optional example)
If you use this model in a report:
```
Author. (2025). 20-Hour Water Level Forecasting Model (Rewaghat Station) [Software].
```

### Disclaimer
Model accuracy may degrade under extreme hydrological events (flash floods, anomalous rainfall). Always pair automated predictions with domain expert review for operational decisions.

---
Feel free to adapt this README as the codebase evolves beyond the notebook prototype stage.


# cwc-waterlevel-forecast-xgboost
XGBoost regression predicts Rewaghat water levels 20h ahead using Chatia data &amp; rainfall. Features: lags, rolling stats, cyclical time, rainfall agg. Custom acc (±0.15m), tuned via RandomizedSearchCV, seasonal eval, recency weighting. High accuracy in stable year. Aids Central Water Commission flood warning &amp; management.
