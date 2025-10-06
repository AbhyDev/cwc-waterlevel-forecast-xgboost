## 20-Hour Water Level Forecasting using XGBoost and LSTM Models

[cite_start]This project develops and compares machine learning models for forecasting water levels at the Rewaghat station 20 hours in advance[cite: 17, 52]. [cite_start]The primary recommended model is an optimized XGBoost regressor, which is benchmarked against three distinct LSTM architectures[cite: 163, 169]. [cite_start]The models are trained on historical gauge data from multiple base stations (Chatia, Rewaghat, Dumariaghat) and aggregated rainfall records[cite: 19, 112, 126].

[cite_start]The final XGBoost model is recommended for operational use due to its high accuracy, which is comparable to the best-performing LSTMs, and its superior computational efficiency[cite: 159, 171].

---

### 📂 Repository Contents

* `XGBoostmodel.ipynb`: End-to-end workflow for the final, recommended XGBoost model. Includes feature engineering, training, and evaluation.
* [cite_start]`LSTMseq-to-vec.ipynb`: Implementation of an LSTM Sequence-to-Vector (Single-Step) model that predicts the 20th hour directly[cite: 128].
* [cite_start]`LSTM_seq-to-vector_multistep.ipynb`: An LSTM Sequence-to-Vector (Multi-Step) model that predicts all 20 future hours simultaneously[cite: 131]. [cite_start]This architecture replicates the CWC's original methodology[cite: 131].
* [cite_start]`LSTMseq-toseq.ipynb`: A true Sequence-to-Sequence (Encoder-Decoder) LSTM model that generates predictions autoregressively[cite: 134].
* Excel data files: `Chatia_*.xlsx`, `Rewaghat_*.xlsx`, `Dumariaghat_data.xlsx`, `rainfall_data.xlsx`.
* `README.md`: This project overview.

---

### 🎯 Models & Methodology

[cite_start]All models use historical data from Chatia, Rewaghat, Dumariaghat, and local rainfall to predict the Rewaghat water level 20 hours ahead[cite: 52, 112, 126]. [cite_start]The primary performance metric is **Custom Accuracy**, defined as the percentage of predictions with an absolute error of ≤ 0.15 meters[cite: 73, 76].

#### Final XGBoost Model (Recommended)

[cite_start]This model was selected for its balance of high predictive accuracy and computational efficiency[cite: 83, 159].
* **Target Variable**: `Rewaghat_20h_future`.
* [cite_start]**Rich Feature Engineering**[cite: 113]:
    * [cite_start]**Lags (1–20 hours)** for Chatia, Rewaghat, and Dumariaghat water levels[cite: 115].
    * [cite_start]**Rolling Stats (mean, std, max)** over 3, 6, 12, and 24-hour windows for all stations[cite: 116].
    * [cite_start]**Rainfall Aggregates**: Rolling sums (6h, 12h, 24h, 48h, 7d), plus 24h max & mean, and hours since last rain[cite: 119].
    * [cite_start]**Temporal Encodings**: Hour, month, day-of-year with sin/cos cyclic signals[cite: 118].
    * [cite_start]**Rate of Change Deltas**: 1-hour and 6-hour differences for all station levels[cite: 117].
* **Sample Weighting**: Emphasizes more recent years during training.

#### Comparative LSTM Models

[cite_start]Three LSTM architectures were developed to benchmark against the XGBoost solution and explore deep learning approaches[cite: 22, 126].
* [cite_start]**LSTM Sequence-to-Vector (Single-Step)**: Uses a 48-hour input sequence to predict only the target 20th hour[cite: 128, 129]. [cite_start]It is the fastest LSTM architecture for this specific task[cite: 130].
* [cite_start]**LSTM Sequence-to-Vector (Multi-Step)**: Takes a 48-hour input and uses the LSTM's final hidden state to predict all 20 future hours simultaneously via a dense output layer[cite: 132].
* [cite_start]**LSTM Sequence-to-Sequence (Autoregressive)**: A true encoder-decoder model that generates the forecast one step at a time, using its own previous prediction as input for the next step[cite: 135].

---

### 📊 Performance Summary

[cite_start]All models were evaluated on the same test set (the most recent year of data)[cite: 70, 142]. [cite_start]The results confirm that the XGBoost model performs on par with the best LSTM variants while being significantly faster to train[cite: 147, 149].

| Model Architecture | Full Year Accuracy | Monsoon (Jun-Oct) Accuracy | Dry Season (Nov-May) Accuracy | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Final XGBoost** | [cite_start]**88.15%** [cite: 144] | [cite_start]**74.29%** [cite: 144] | [cite_start]**97.94%** [cite: 144] | [cite_start]**Recommended model; computationally efficient.** [cite: 144, 159] |
| LSTM Seq-to-Vec (Single) | [cite_start]88.76% [cite: 144] | [cite_start]74.92% [cite: 144] | [cite_start]98.81% [cite: 144] | Predicts only the 20th hour. [cite_start]Highest accuracy. [cite: 144] |
| LSTM Seq-to-Vec (Multi) | [cite_start]88.70% [cite: 144] | [cite_start]74.75% [cite: 144] | [cite_start]98.83% [cite: 144] | [cite_start]Replicates CWC's original method. [cite: 144] |
| LSTM Seq-to-Seq (Auto) | [cite_start]85.63% [cite: 144] | [cite_start]68.23% [cite: 144] | [cite_start]98.26% [cite: 144] | [cite_start]Weakest performer, prone to error propagation. [cite: 144, 154] |

**Analysis**:
* [cite_start]Forecasting during the non-monsoon "Dry Season" is highly accurate (~98%) across all top models[cite: 156].
* [cite_start]The "Monsoon Season" remains the primary challenge (~74% accuracy) due to volatile conditions caused by heavy rainfall[cite: 156, 157].
* [cite_start]The autoregressive Seq-to-Seq LSTM performed worst, likely due to error propagation where inaccuracies in early predictions are amplified in later steps[cite: 153, 154].

---

### 💾 Data Split Strategy

[cite_start]A time-ordered split is used across all notebooks to prevent data leakage and simulate a real-world forecasting scenario[cite: 67]:
* [cite_start]**Train Set**: All data except the last 2 years[cite: 68].
* [cite_start]**Validation Set**: The second-to-last year of data[cite: 69].
* [cite_start]**Test Set**: The most recent 1 year of data[cite: 70].

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