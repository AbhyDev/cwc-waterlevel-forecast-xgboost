This project was developed under the Central Water Commission under the title **Flood Forecasting and Meteorological Analysis for Water Resources Management using Machine Learning**.

## 20-Hour Water Level Forecasting using XGBoost and LSTM Models

This project develops and compares machine learning models for forecasting water levels at the Rewaghat station 20 hours in advance. The primary recommended model is an optimized XGBoost regressor, which is benchmarked against three distinct LSTM architectures. The models are trained on historical gauge data from multiple base stations (Chatia, Rewaghat, Dumariaghat) and aggregated rainfall records.

The final XGBoost model is recommended for operational use due to its high accuracy, which is comparable to the best-performing LSTMs, and its superior computational efficiency.

---

### 📂 Repository Contents

* `XGBoostmodel.ipynb`: End-to-end workflow for the final, recommended XGBoost model. Includes feature engineering, training, and evaluation.
* `LSTMseq-to-vec.ipynb`: Implementation of an LSTM Sequence-to-Vector (Single-Step) model that predicts the 20th hour directly.
* `LSTM_seq-to-vector_multistep.ipynb`: An LSTM Sequence-to-Vector (Multi-Step) model that predicts all 20 future hours simultaneously. This architecture replicates the CWC's original methodology.
* `LSTMseq-toseq.ipynb`: A true Sequence-to-Sequence (Encoder-Decoder) LSTM model that generates predictions autoregressively.
* Excel data files: `Chatia_*.xlsx`, `Rewaghat_*.xlsx`, `Dumariaghat_data.xlsx`, `rainfall_data.xlsx`.
* `README.md`: This project overview.

---

### 🎯 Models & Methodology

All models use historical data from Chatia, Rewaghat, Dumariaghat, and local rainfall to predict the Rewaghat water level 20 hours ahead. The primary performance metric is **Custom Accuracy**, defined as the percentage of predictions with an absolute error of ≤ 0.15 meters.

#### Final XGBoost Model (Recommended)

This model was selected for its balance of high predictive accuracy and computational efficiency.
* **Target Variable**: `Rewaghat_20h_future`.
* **Rich Feature Engineering**:
    * **Lags (1–20 hours)** for Chatia, Rewaghat, and Dumariaghat water levels.
    * **Rolling Stats (mean, std, max)** over 3, 6, 12, and 24-hour windows for all stations.
    * **Rainfall Aggregates**: Rolling sums (6h, 12h, 24h, 48h, 7d), plus 24h max & mean, and hours since last rain.
    * **Temporal Encodings**: Hour, month, day-of-year with sin/cos cyclic signals.
    * **Rate of Change Deltas**: 1-hour and 6-hour differences for all station levels.
* **Sample Weighting**: Emphasizes more recent years during training.

#### Comparative LSTM Models

Three LSTM architectures were developed to benchmark against the XGBoost solution and explore deep learning approaches.
* **LSTM Sequence-to-Vector (Single-Step)**: Uses a 48-hour input sequence to predict only the target 20th hour. It is the fastest LSTM architecture for this specific task.
* **LSTM Sequence-to-Vector (Multi-Step)**: Takes a 48-hour input and uses the LSTM's final hidden state to predict all 20 future hours simultaneously via a dense output layer.
* **LSTM Sequence-to-Sequence (Autoregressive)**: A true encoder-decoder model that generates the forecast one step at a time, using its own previous prediction as input for the next step.

---

### 📊 Performance Summary

All models were evaluated on the same test set (the most recent year of data). The results confirm that the XGBoost model performs on par with the best LSTM variants while being significantly faster to train.

| Model Architecture | Full Year Accuracy | Monsoon (Jun-Oct) Accuracy | Dry Season (Nov-May) Accuracy | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Final XGBoost** | **88.15%** | **74.29%** | **97.94%** | **Recommended model; computationally efficient.** |
| LSTM Seq-to-Vec (Single) | 88.76% | 74.92% | 98.81% | Predicts only the 20th hour. Highest accuracy. |
| LSTM Seq-to-Vec (Multi) | 88.70% | 74.75% | 98.83% | Replicates CWC's original method. |
| LSTM Seq-to-Seq (Auto) | 85.63% | 68.23% | 98.26% | Weakest performer, prone to error propagation. |

**Analysis**:
* Forecasting during the non-monsoon "Dry Season" is highly accurate (~98%) across all top models.
* The "Monsoon Season" remains the primary challenge (~74% accuracy) due to volatile conditions caused by heavy rainfall.
* The autoregressive Seq-to-Seq LSTM performed worst, likely due to error propagation where inaccuracies in early predictions are amplified in later steps.

---

### 💾 Data Split Strategy

A time-ordered split is used across all notebooks to prevent data leakage and simulate a real-world forecasting scenario:
* **Train Set**: All data except the last 2 years.
* **Validation Set**: The second-to-last year of data.
* **Test Set**: The most recent 1 year of data.

---

### 🏗️ Model Artifacts

* **XGBoost**:
    * `best_xgb_params_20hr.pkl`: Best hyperparameters from randomized search.
    * `xgb_rewaghat_model_bundle_20hr.pkl`: A dictionary containing the trained `XGBRegressor` model and the required feature names.
* **LSTM**:
    * `best_lstm_comparison_model.pth`: Trained weights for the Seq-to-Vec (Single-Step) model.
    * `best_lstm_seq2seq_model.pth`: Trained weights for the Seq-to-Vec (Multi-Step) model.
    * `best_true_seq2seq_model.pth`: Trained weights for the Seq-to-Seq model.

---

### 🚀 Quick Start

#### Install Environment
The project uses `uv` for environment management, but `pip` with a virtual environment is also supported.

```bash
# Using uv (recommended)
pip install uv
uv sync

# Or, using venv + pip
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# source .venv/Scripts/activate # On Windows
pip install pandas numpy xgboost scikit-learn matplotlib joblib openpyxl torch