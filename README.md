## 20-Hour Water Level Forecasting (Rewaghat Station)

This project predicts Rewaghat station water levels 20 hours ahead using an XGBoost regression model trained on historical gauge data (Chatia & Rewaghat) and aggregated rainfall records.

📂 Repository Contents

`20_hours_predicting_final.ipynb` – full end-to-end workflow: data merging, feature engineering, model tuning, training, evaluation, and seasonal analysis.

Excel data – `Chatia_*.xlsx`, `Rewaghat_*.xlsx`, `rainfall_data.xlsx`.

`pyproject.toml` – project metadata and dependencies.

`main.py` – placeholder entry point (extend for CLI or service integration).

`uv.lock` – environment lockfile for reproducibility.

🎯 Key Features

20-hour ahead forecasting target: `Rewaghat_20h_future`.

Rich feature engineering:

Lags (1–20) for Chatia & Rewaghat levels.

Rolling stats (mean, std, max) over 3h / 6h / 12h / 24h.

Rainfall aggregates: 6h / 12h / 24h / 48h / 7d sums, plus 24h max & mean, lag, hours since last rain.

Temporal encodings: hour, month, day-of-year + sin/cos cyclic signals.

Short- and medium-term change deltas (1h, 6h).

Sample weighting emphasizes recent years.

Custom accuracy metric: fraction of predictions within ±0.15 m.

Seasonal evaluation: monsoon (Jun–Oct) vs dry season (Nov–May).

📊 Data Split Strategy

Time-ordered split to avoid leakage:

Train: all data except last 2 years.

Validation: year −2 to year −1.

Test: most recent 1 year (with monsoon & dry subsets).

🏗️ Model Artifacts

Saved after notebook execution:

`best_xgb_params_20hr.pkl` – best hyperparameters from randomized search.

`xgb_rewaghat_model_bundle_20hr.pkl` – dictionary containing:

```python
{
	"model": XGBRegressor(...),
	"feature_names": [... feature column order ...]
}
```

🚀 Quick Start
Install Environment

Option A (with uv):

```bash
pip install uv --upgrade
uv sync
```

Option B (classic venv + pip):

```bash
python -m venv .venv
source .venv/Scripts/activate   # on Windows (Git Bash)
pip install -e .
```

Launch Notebook

```bash
python -m ipykernel install --user --name cwc20
```

Open `20_hours_predicting_final.ipynb` and run all cells.

🔍 Using the Trained Model

```python
import joblib
import pandas as pd

bundle = joblib.load("xgb_rewaghat_model_bundle_20hr.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]

# df must be preprocessed with the same pipeline
X = your_feature_engineered_dataframe[feature_names]
pred_20h = model.predict(X)
```

📦 Project Structure
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

🔮 Next Steps

Refactor feature engineering into `src/` module with tests.

Add CLI (`python -m cwc_forecast predict --input latest.xlsx`).

Integrate drift monitoring & retraining pipeline.

Automate rainfall/station ingestion scripts.

Add `.gitignore` for large artifacts and venvs.

Suggested `.gitignore`:

```gitignore
__pycache__/
.venv/
*.pyc
.ipynb_checkpoints/
*.log
*.pkl
# Uncomment to exclude raw spreadsheets:
# *.xlsx
```

⚠️ Disclaimer

This model is a research prototype. Accuracy may degrade under extreme hydrological events (flash floods, anomalous rainfall). Always pair predictions with expert hydrologist review before operational use.

🛠️ Tech Stack

Python: 3.13+

Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, joblib, openpyxl

