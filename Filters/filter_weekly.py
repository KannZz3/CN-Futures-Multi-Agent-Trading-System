# -*- coding: utf-8 -*-
"""Auto-split from the original single-contract notebooks.
Core formulas, thresholds, and function bodies are kept unchanged.
"""

import numpy as np
import pandas as pd

def get_test_week_info(test_df: pd.DataFrame, weekly_df: pd.DataFrame):
    if not isinstance(test_df.index, pd.DatetimeIndex):
        raise ValueError("test_df.index must be a DatetimeIndex")

    if not isinstance(weekly_df.index, pd.DatetimeIndex):
        raise ValueError("weekly_df.index must be a DatetimeIndex")

    test_df = test_df.sort_index()
    weekly_df = weekly_df.sort_index()

    first_test_date = test_df.index.min()

    weekly_after = weekly_df[weekly_df.index >= first_test_date]
    if weekly_after.empty:
        raise ValueError(f"No weekly data found in weekly_df with index >= {first_test_date}")

    test_start_week = weekly_after.index[0]

    return first_test_date, test_start_week

def fixed_train_zscore(
    series: pd.Series,
    train_mask,
    name: str,
    min_train_obs: int = 20,
):
    train_s = series.loc[train_mask].dropna()

    if train_s.empty:
        raise ValueError(f"{name}: train sample is empty")

    if len(train_s) < min_train_obs:
        raise ValueError(
            f"{name}: train sample too small: {len(train_s)} < {min_train_obs}"
        )

    mu = train_s.mean()
    sd = train_s.std(ddof=1)

    if pd.isna(sd) or sd == 0:
        raise ValueError(f"{name}: train std invalid")

    z = (series - mu) / sd

    return z, mu, sd

def build_weekly_filters_no_leak(
    weekly_df: pd.DataFrame,
    test_df: pd.DataFrame,
    price_col: str = "close",
    hold_col: str = "hold",
    vol_window: int = 5,
    trend_fast: int = 3,
    trend_slow: int = 6,
    crowd_lookback: int = 4,
    vol_z_low: float = -0.5,
    vol_z_high: float = 1,
    trend_z_th: float = 0.5,
    crowd_hold_z_th: float = 1,
    crowd_price_z_th: float = 1,
):
    if not isinstance(weekly_df.index, pd.DatetimeIndex):
        raise ValueError("weekly_df.index must be a DatetimeIndex")

    if not isinstance(test_df.index, pd.DatetimeIndex):
        raise ValueError("test_df.index must be a DatetimeIndex")

    if price_col not in weekly_df.columns:
        raise ValueError(f"weekly_df missing price column: {price_col}")

    if hold_col not in weekly_df.columns:
        raise ValueError(f"weekly_df missing hold column: {hold_col}")

    first_test_date, test_start_week = get_test_week_info(test_df, weekly_df)

    w = weekly_df.sort_index().copy()
    w = w[~w.index.duplicated(keep="last")]

    # Weekly training sample:
    # strictly before the week containing the first test date
    train_mask = w.index < test_start_week

    price = w[price_col].astype(float)
    hold = w[hold_col].astype(float)

    # ============================================================
    # 1) Raw weekly indicators
    # ============================================================
    wk_ret = price.pct_change()

    wk_vol = wk_ret.rolling(
        window=vol_window,
        min_periods=max(2, vol_window // 2)
    ).std()

    ma_fast = price.rolling(
        window=trend_fast,
        min_periods=max(2, trend_fast // 2)
    ).mean()

    ma_slow = price.rolling(
        window=trend_slow,
        min_periods=max(2, trend_slow // 2)
    ).mean()

    trend_raw = ma_fast - ma_slow

    hold_chg = hold.diff()
    hold_chg_L = hold - hold.shift(crowd_lookback)
    price_chg_L = price / price.shift(crowd_lookback) - 1.0

    # ============================================================
    # 2) Estimate all normalization parameters
    #    from weekly TRAIN data only
    # ============================================================
    wk_ret_z, ret_mu, ret_sd = fixed_train_zscore(
        wk_ret, train_mask, "wk_ret"
    )

    wk_vol_z, vol_mu, vol_sd = fixed_train_zscore(
        wk_vol, train_mask, "wk_vol"
    )

    wk_trend_z, trend_mu, trend_sd = fixed_train_zscore(
        trend_raw, train_mask, "wk_trend_raw"
    )

    wk_hold_chg_L_z, hold_L_mu, hold_L_sd = fixed_train_zscore(
        hold_chg_L, train_mask, "wk_hold_chg_L"
    )

    wk_price_chg_L_z, price_L_mu, price_L_sd = fixed_train_zscore(
        price_chg_L, train_mask, "wk_price_chg_L"
    )

    # ============================================================
    # 3) Construct weekly filters
    # ============================================================
    wk_vol_regime = pd.Series(np.nan, index=w.index, dtype=float)
    wk_vol_regime[wk_vol_z <= vol_z_low] = 0.0
    wk_vol_regime[(wk_vol_z > vol_z_low) & (wk_vol_z < vol_z_high)] = 1.0
    wk_vol_regime[wk_vol_z >= vol_z_high] = 2.0

    wk_trend_sign = pd.Series(0.0, index=w.index, dtype=float)
    wk_trend_sign[wk_trend_z > trend_z_th] = 1.0
    wk_trend_sign[wk_trend_z < -trend_z_th] = -1.0

    # Long crowdedness:
    # significant position increase + significant price rally
    wk_crowded_long = (
        (wk_hold_chg_L_z > crowd_hold_z_th) &
        (wk_price_chg_L_z > crowd_price_z_th)
    ).astype(float)

    # Short crowdedness:
    # significant position increase + significant price decline
    wk_crowded_short = (
        (wk_hold_chg_L_z > crowd_hold_z_th) &
        (wk_price_chg_L_z < -crowd_price_z_th)
    ).astype(float)

    # ============================================================
    # 4) Keep output column names aligned with backtest_money
    # ============================================================
    weekly_filters = pd.DataFrame(index=w.index)

    weekly_filters["wk_ret"] = wk_ret
    weekly_filters["wk_ret_z"] = wk_ret_z

    weekly_filters["wk_vol"] = wk_vol
    weekly_filters["wk_vol_z"] = wk_vol_z
    weekly_filters["wk_vol_regime"] = wk_vol_regime

    weekly_filters["wk_ma_fast"] = ma_fast
    weekly_filters["wk_ma_slow"] = ma_slow
    weekly_filters["wk_trend_raw"] = trend_raw
    weekly_filters["wk_trend_z"] = wk_trend_z
    weekly_filters["wk_trend_sign"] = wk_trend_sign

    weekly_filters["wk_hold_chg"] = hold_chg
    weekly_filters["wk_hold_chg_L"] = hold_chg_L
    weekly_filters["wk_hold_chg_L_z"] = wk_hold_chg_L_z

    weekly_filters["wk_price_chg_L"] = price_chg_L
    weekly_filters["wk_price_chg_L_z"] = wk_price_chg_L_z

    weekly_filters["wk_crowded_long"] = wk_crowded_long
    weekly_filters["wk_crowded_short"] = wk_crowded_short

    meta = {
        "first_test_date": first_test_date,
        "test_start_week": test_start_week,
        "ret_mu": ret_mu,
        "ret_sd": ret_sd,
        "vol_mu": vol_mu,
        "vol_sd": vol_sd,
        "trend_mu": trend_mu,
        "trend_sd": trend_sd,
        "hold_L_mu": hold_L_mu,
        "hold_L_sd": hold_L_sd,
        "price_L_mu": price_L_mu,
        "price_L_sd": price_L_sd,
        "daily_alignment": "each week uses previous week's completed weekly filter",
    }

    return weekly_filters, meta

def attach_weekly_filters_to_daily_no_leak(
    daily_df: pd.DataFrame,
    weekly_filters: pd.DataFrame,
):
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        raise ValueError("daily_df.index must be a DatetimeIndex")

    if not isinstance(weekly_filters.index, pd.DatetimeIndex):
        raise ValueError("weekly_filters.index must be a DatetimeIndex")

    daily = daily_df.sort_index().copy()
    weekly = weekly_filters.sort_index().copy()

    daily = daily[~daily.index.duplicated(keep="last")]
    weekly = weekly[~weekly.index.duplicated(keep="last")]

    old_wk_cols = [c for c in daily.columns if c.startswith("wk_")]
    if old_wk_cols:
        daily = daily.drop(columns=old_wk_cols)

    weekly_lagged = weekly.shift(1)

    week_dates = daily.index.to_period("W-FRI").to_timestamp("W-FRI")

    weekly_for_daily = weekly_lagged.reindex(week_dates)
    weekly_for_daily.index = daily.index

    out = pd.concat([daily, weekly_for_daily], axis=1)

    return out
