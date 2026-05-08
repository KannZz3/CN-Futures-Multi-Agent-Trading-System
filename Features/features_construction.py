# -*- coding: utf-8 -*-
"""Auto-split from the original single-contract notebooks.
Core formulas, thresholds, and function bodies are kept unchanged.
"""

import numpy as np
import pandas as pd

def load_main_csv(csv_path):

    df = pd.read_csv(csv_path)

    # 1) Build datetime index from `date`
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.sort_values("datetime").set_index("datetime")

    # 2) Use adjusted OHLC columns (adj_*) and keep volume / hold as is
    data = pd.DataFrame(index=df.index)
    data["open"]   = df["adj_open"].astype(float)
    data["high"]   = df["adj_high"].astype(float)
    data["low"]    = df["adj_low"].astype(float)
    data["close"]  = df["adj_close"].astype(float)
    data["volume"] = df["volume"].astype(float)
    data["hold"]   = df["hold"].astype(float)
    return data

def build_daily_features(
    df,
    mom_windows=(2, 4, 8, 16),
    vol_windows=(2, 4, 8, 16, 32),
    ma_windows=(2, 4, 8, 16, 32),
    vol_ma_windows=(3, 5, 10, 20),
    hold_ma_windows=(3, 5, 10, 20),
    donchian_windows=(10, 20, 55),
    osc_windows=(7, 14, 28),
):
    data = df.copy()
    eps = 1e-9

    # =========================
    # 1. Single-bar return & candlestick shape
    # =========================
    data["ret_1"] = data["close"].pct_change()

    data["body"] = data["close"] - data["open"]
    data["range"] = data["high"] - data["low"]
    data["upper_shadow"] = data["high"] - data[["open", "close"]].max(axis=1)
    data["lower_shadow"] = data[["open", "close"]].min(axis=1) - data["low"]

    range_safe = data["range"].replace(0, np.nan)

    data["body_pct"] = data["body"] / range_safe
    data["upper_shadow_pct"] = data["upper_shadow"] / range_safe
    data["lower_shadow_pct"] = data["lower_shadow"] / range_safe

    data["is_bull"] = (data["close"] > data["open"]).astype(int)
    data["is_bear"] = (data["close"] < data["open"]).astype(int)

    prev_close = data["close"].shift(1)

    data["gap_open_ret"] = data["open"] / prev_close.replace(0, np.nan) - 1
    data["intraday_ret"] = data["close"] / data["open"].replace(0, np.nan) - 1
    data["close_position_in_bar"] = (
        (data["close"] - data["low"]) / range_safe
    ).fillna(0.5)

    data["gap_intraday_div"] = (
        data["gap_open_ret"] - data["intraday_ret"]
    )

    # =========================
    # 2. Momentum, volatility & moving-average factors
    # =========================
    for n in mom_windows:
        col = f"mom_{n}"

        data[col] = data["close"].pct_change(n)

        data[f"{col}_slope"] = data[col].diff()
        data[f"{col}_slope_3"] = data[col].diff(3)
        data[f"{col}_slope_5"] = data[col].diff(5)
        data[f"{col}_slope_10"] = data[col].diff(10)
        data[f"{col}_slope_25"] = data[col].diff(25)

        data[f"{col}_acc"] = data[col].diff().diff()

        data[f"mom_{n}_angle"] = np.arctan(
            data[col].diff() / (data[col].shift(1).abs() + eps)
        )

    for n in vol_windows:
        col = f"vol_{n}"

        data[col] = data["ret_1"].rolling(n).std()

        data[f"{col}_slope"] = data[col].diff()
        data[f"{col}_slope_3"] = data[col].diff(3)
        data[f"{col}_slope_5"] = data[col].diff(5)
        data[f"{col}_slope_10"] = data[col].diff(10)
        data[f"{col}_slope_25"] = data[col].diff(25)

        data[f"{col}_acc"] = data[col].diff().diff()

    # =========================
    # Weighted moving averages (WMA)
    # =========================
    data["wma_5"] = data["close"].rolling(5).apply(
        lambda x: np.dot(x, np.arange(1, 6)) / np.sum(np.arange(1, 6)),
        raw=True
    )

    data["wma_10"] = data["close"].rolling(10).apply(
        lambda x: np.dot(x, np.arange(1, 11)) / np.sum(np.arange(1, 11)),
        raw=True
    )

    data["close_wma_5"] = data["close"] / data["wma_5"] - 1
    data["close_wma_10"] = data["close"] / data["wma_10"] - 1

    for n in ma_windows:
        ma_col = f"ma_{n}"

        data[ma_col] = data["close"].rolling(n).mean()

        data[f"close_ma_{n}"] = (
            data["close"] / data[ma_col] - 1
        )

        data[f"{ma_col}_slope"] = data[ma_col].pct_change()
        data[f"{ma_col}_slope_3"] = data[ma_col].pct_change(3)
        data[f"{ma_col}_slope_5"] = data[ma_col].pct_change(5)
        data[f"{ma_col}_slope_10"] = data[ma_col].pct_change(10)
        data[f"{ma_col}_slope_25"] = data[ma_col].pct_change(25)

        data[f"{ma_col}_acc"] = data[ma_col].diff().diff()

        data[f"ma_{n}_angle"] = np.arctan(
            data[ma_col].diff() / (data[ma_col].shift(1).abs() + eps)
        )

    # =========================
    # 2B. Donchian breakout factors
    # =========================
    for n in donchian_windows:
        roll_high = data["high"].rolling(n).max()
        roll_low = data["low"].rolling(n).min()

        width = (roll_high - roll_low).replace(0, np.nan)

        data[f"donchian_width_{n}"] = (
            width / data["close"].replace(0, np.nan)
        )

        data[f"donchian_pos_{n}"] = (
            (data["close"] - roll_low) / width
        )

        data[f"breakout_up_{n}"] = (
            data["close"] > roll_high.shift(1)
        ).astype(int)

        data[f"breakout_down_{n}"] = (
            data["close"] < roll_low.shift(1)
        ).astype(int)

    # =========================
    # 3. Volume & open-interest factors
    # =========================
    log_volume = np.log1p(data["volume"].clip(lower=0))
    log_hold = np.log1p(data["hold"].clip(lower=0))

    data["vol_ret_1"] = log_volume.diff()
    data["vol_ret_3"] = log_volume.diff(3)
    data["vol_ret_5"] = log_volume.diff(5)
    data["vol_ret_10"] = log_volume.diff(10)
    data["vol_ret_15"] = log_volume.diff(15)
    data["vol_ret_20"] = log_volume.diff(20)
    data["vol_ret_25"] = log_volume.diff(25)

    for n in vol_ma_windows:
        ma_col = f"vol_ma_{n}"

        data[ma_col] = data["volume"].rolling(n).mean()

        data[f"vol_ma_bias_{n}"] = (
            data["volume"] / data[ma_col].replace(0, np.nan) - 1
        )

    for n in vol_ma_windows:
        hold_ma_col = f"hold_ma_for_volume_ratio_{n}"

        data[hold_ma_col] = data["hold"].rolling(n).mean()

        data[f"volume_to_hold_ma_{n}"] = (
            data["volume"] / data[hold_ma_col].replace(0, np.nan)
        )

    for n in vol_ma_windows:
        data[f"vol_vol_{n}"] = (
            data["vol_ret_1"].rolling(n).std()
        )

    data["vol_price_div_1"] = (
        data["vol_ret_1"] - data["ret_1"]
    )

    data["vol_price_div_3"] = (
        data["vol_ret_3"] - data["close"].pct_change(3)
    )

    data["vol_price_div_5"] = (
        data["vol_ret_5"] - data["close"].pct_change(5)
    )

    data["vol_price_div_10"] = (
        data["vol_ret_10"] - data["close"].pct_change(10)
    )

    data["vol_price_div_15"] = (
        data["vol_ret_15"] - data["close"].pct_change(15)
    )

    # =========================
    # 4. Hold / open-interest factors
    # =========================
    data["hold_ret_1"] = log_hold.diff()
    data["hold_ret_3"] = log_hold.diff(3)
    data["hold_ret_5"] = log_hold.diff(5)
    data["hold_ret_10"] = log_hold.diff(10)
    data["hold_ret_15"] = log_hold.diff(15)
    data["hold_ret_20"] = log_hold.diff(20)

    for n in hold_ma_windows:
        ma_col = f"hold_ma_{n}"

        data[ma_col] = data["hold"].rolling(n).mean()

        data[f"hold_ma_bias_{n}"] = (
            data["hold"] / data[ma_col].replace(0, np.nan) - 1
        )

    for n in hold_ma_windows:
        data[f"hold_vol_{n}"] = (
            data["hold_ret_1"].rolling(n).std()
        )

    data["price_oi_div_1"] = (
        data["ret_1"] - data["hold_ret_1"]
    )

    data["price_oi_div_3"] = (
        data["close"].pct_change(3) - data["hold_ret_3"]
    )

    data["price_oi_div_5"] = (
        data["close"].pct_change(5) - data["hold_ret_5"]
    )

    data["price_oi_div_10"] = (
        data["close"].pct_change(10) - data["hold_ret_10"]
    )

    data["price_oi_div_15"] = (
        data["close"].pct_change(15) - data["hold_ret_15"]
    )

    data["volume_hold_ratio"] = (
        data["volume"] / (data["hold"] + eps)
    )

    data["vp_ratio_5"] = (
        data["vol_ret_1"].rolling(5).corr(data["ret_1"])
    )

    data["vp_ratio_10"] = (
        data["vol_ret_1"].rolling(10).corr(data["ret_1"])
    )

    data["hp_ratio_5"] = (
        data["hold_ret_1"].rolling(5).corr(data["ret_1"])
    )

    data["hp_ratio_10"] = (
        data["hold_ret_1"].rolling(10).corr(data["ret_1"])
    )

    # =========================
    # 5. Combined price-volume-OI interaction factors
    # =========================
    data["price_vol_confirm"] = (
        np.sign(data["ret_1"]) * data["vol_ret_1"]
    )

    data["price_oi_trend"] = (
        np.sign(data["ret_1"]) * data["hold_ret_1"]
    )

    data["absorption_ratio"] = (
        np.log1p(data["volume"])
        - np.log(data["body"].abs() + 1e-3)
    )

    data["effort_result_body_abs"] = (
        np.log1p(data["volume"]) * data["body"].abs()
    )

    data["pv_corr_5"] = (
        data["ret_1"].rolling(5).corr(data["vol_ret_1"])
    )

    data["po_corr_5"] = (
        data["ret_1"].rolling(5).corr(data["hold_ret_1"])
    )

    data["vo_corr_5"] = (
        data["vol_ret_1"].rolling(5).corr(data["hold_ret_1"])
    )

    # =========================
    # 6. Multi-window RSI
    # =========================
    delta = data["close"].diff()

    for n in osc_windows:
        gain = delta.where(delta > 0, 0).rolling(n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(n).mean()

        rs = gain / loss.replace(0, np.nan)

        rsi = 100 - 100 / (1 + rs)

        rsi = rsi.mask((loss == 0) & (gain > 0), 100)
        rsi = rsi.mask((gain == 0) & (loss > 0), 0)
        rsi = rsi.mask((gain == 0) & (loss == 0), 50)

        data[f"RSI_{n}"] = rsi
        data[f"RSI_{n}_slope"] = data[f"RSI_{n}"].diff()
        data[f"RSI_{n}_midline"] = data[f"RSI_{n}"] - 50

    # =========================
    # 7. Multi-window stochastic oscillator / KDJ
    # =========================
    for n in osc_windows:
        low_n = data["low"].rolling(n).min()
        high_n = data["high"].rolling(n).max()

        denom = (high_n - low_n).replace(0, np.nan)

        data[f"stoch_k_{n}"] = (
            (data["close"] - low_n) / denom * 100
        )

        data[f"stoch_d_{n}"] = (
            data[f"stoch_k_{n}"].rolling(3).mean()
        )

        data[f"stoch_j_{n}"] = (
            3 * data[f"stoch_k_{n}"]
            - 2 * data[f"stoch_d_{n}"]
        )

        data[f"stoch_kd_diff_{n}"] = (
            data[f"stoch_k_{n}"]
            - data[f"stoch_d_{n}"]
        )

    # Keep legacy column name to avoid breaking
    # downstream feature_cols or older code references
    data["stochastic_14"] = data["stoch_k_14"]

    data = data.replace([np.inf, -np.inf], np.nan)

    return data
