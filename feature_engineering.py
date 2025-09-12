import numpy as np
import pandas as pd

def create_features(df, extended: bool = False, max_lag: int = 72):
    """
    Create time-series features (lags, rollings, calendar, rainfall, deltas) without causing
    DataFrame fragmentation. All historical features are shifted to prevent leakage.

    Params:
    - extended: when True, include longer lags/rollings up to `max_lag` (useful for monsoon).
    - max_lag: maximum lag to compute when `extended=True` (default 72h).
    """
    base = df.copy()

    # Containers for batched feature creation
    feats = {}

    # --- Target: 20 hours into the future ---
    target = base['Rewaghat'].shift(-20)

    # Base lags (ensure backwards compatibility)
    for i in range(1, 21):
        feats[f'Chatia_lag_{i}'] = base['Chatia'].shift(i)
        feats[f'Rewaghat_lag_{i}'] = base['Rewaghat'].shift(i)

    # Extended longer lags (e.g., to 72h) when requested
    if extended and max_lag > 20:
        for i in range(21, max_lag + 1):
            feats[f'Chatia_lag_{i}'] = base['Chatia'].shift(i)
            feats[f'Rewaghat_lag_{i}'] = base['Rewaghat'].shift(i)

    # Rolling windows (base)
    for window in [3, 6, 12, 24]:
        c_shift = base['Chatia'].shift(1)
        r_shift = base['Rewaghat'].shift(1)
        feats[f'Chatia_rolling_mean_{window}'] = c_shift.rolling(window=window).mean()
        feats[f'Chatia_rolling_std_{window}']  = c_shift.rolling(window=window).std()
        feats[f'Chatia_rolling_max_{window}']  = c_shift.rolling(window=window).max()
        feats[f'Rewaghat_rolling_mean_{window}'] = r_shift.rolling(window=window).mean()
        feats[f'Rewaghat_rolling_std_{window}']  = r_shift.rolling(window=window).std()
        feats[f'Rewaghat_rolling_max_{window}']  = r_shift.rolling(window=window).max()

    # Extended rollings (larger windows helpful in monsoon)
    if extended:
        for window in [36, 48, 72]:
            c_shift = base['Chatia'].shift(1)
            r_shift = base['Rewaghat'].shift(1)
            feats[f'Chatia_rolling_mean_{window}'] = c_shift.rolling(window=window).mean()
            feats[f'Chatia_rolling_std_{window}']  = c_shift.rolling(window=window).std()
            feats[f'Chatia_rolling_max_{window}']  = c_shift.rolling(window=window).max()
            feats[f'Rewaghat_rolling_mean_{window}'] = r_shift.rolling(window=window).mean()
            feats[f'Rewaghat_rolling_std_{window}']  = r_shift.rolling(window=window).std()
            feats[f'Rewaghat_rolling_max_{window}']  = r_shift.rolling(window=window).max()

        # Targeted upstream travel-time signals centered ~40–48h
        # Mean Chatia levels over lags 40..48 (inclusive)
        chatia_lags_40_48 = [base['Chatia'].shift(k) for k in range(40, 49)]
        feats['Chatia_mean_lag_40_48'] = pd.concat(chatia_lags_40_48, axis=1).mean(axis=1)

        # Upstream-downstream gap at specific lags: Chatia(t-k) - Rewaghat(t-1)
        r_tm1 = base['Rewaghat'].shift(1)
        for k in [36, 40, 44, 48]:
            feats[f'upstream_downstream_gap_{k}h'] = base['Chatia'].shift(k) - r_tm1

        # Exponential moving averages (smoothed state)
        feats['Rewaghat_ewm_24'] = base['Rewaghat'].shift(1).ewm(span=24, adjust=False).mean()
        feats['Chatia_ewm_24'] = base['Chatia'].shift(1).ewm(span=24, adjust=False).mean()

    # Calendar features (vectorized)
    hour = base.index.hour
    month = base.index.month
    feats['hour'] = hour
    feats['month'] = month
    feats['day_of_year'] = base.index.dayofyear
    feats['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    feats['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    feats['month_sin'] = np.sin(2 * np.pi * month / 12)
    feats['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Rainfall summaries
    rf_shift = base['Rainfall'].shift(1)
    feats['rainfall_lag_1'] = rf_shift
    feats['rainfall_3h_sum'] = rf_shift.rolling(window=3).sum()
    feats['rainfall_6h_sum'] = rf_shift.rolling(window=6).sum()
    feats['rainfall_12h_sum'] = rf_shift.rolling(window=12).sum()
    feats['rainfall_24h_sum'] = rf_shift.rolling(window=24).sum()
    feats['rainfall_48h_sum'] = rf_shift.rolling(window=48).sum()
    feats['rainfall_72h_sum'] = rf_shift.rolling(window=72).sum()
    feats['rainfall_7d_sum'] = rf_shift.rolling(window=168).sum()
    feats['rainfall_24h_max'] = rf_shift.rolling(window=24).max()
    feats['rainfall_24h_mean'] = rf_shift.rolling(window=24).mean()
    rain_mask = rf_shift > 0.1
    feats['hours_since_rain'] = rain_mask.groupby(rain_mask.cumsum()).cumcount()

    # Level changes (based on t-1 to avoid leakage)
    feats['Chatia_change_1h'] = base['Chatia'].shift(1).diff(1)
    feats['Rewaghat_change_1h'] = base['Rewaghat'].shift(1).diff(1)
    feats['Chatia_change_6h'] = base['Chatia'].shift(1).diff(6)
    feats['Rewaghat_change_6h'] = base['Rewaghat'].shift(1).diff(6)
    if extended:
        feats['Chatia_change_3h'] = base['Chatia'].shift(1).diff(3)
        feats['Rewaghat_change_3h'] = base['Rewaghat'].shift(1).diff(3)
        feats['Chatia_change_12h'] = base['Chatia'].shift(1).diff(12)
        feats['Rewaghat_change_12h'] = base['Rewaghat'].shift(1).diff(12)

    # Assemble final DataFrame in one go to avoid fragmentation
    features_df = pd.DataFrame(feats, index=base.index)
    out = pd.concat([
        base,  # keep original columns (Chatia, Rewaghat, Rainfall)
        features_df,
        pd.Series(target, name='Rewaghat_20h_future')
    ], axis=1)

    out.dropna(inplace=True)
    return out
