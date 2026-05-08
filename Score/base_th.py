# -*- coding: utf-8 -*-
"""Auto-split from the original single-contract notebooks.
Core formulas, thresholds, and function bodies are kept unchanged.
"""

import numpy as np
import pandas as pd

def choose_base_th_from_train(
    train_df,
    close_col="close",
    base_th_ref=0.01,     # Target cum_ret threshold when daily vol is close to ref_vol
    vol_lookback=50,      # Rolling window length for volatility estimation
    ref_vol=0.01,         # Reference baseline volatility, default 1%
    floor_mult=0.5,       # Lower bound: base_th >= base_th_ref * floor_mult
    ceil_mult=3.0,        # Upper bound: base_th <= base_th_ref * ceil_mult
    winsor_pct=1.0,       # Winsorize both tails of daily returns by winsor_pct%
    min_vol=0.002,        # Lower bound for vol_median, e.g. 0.2%
    max_vol=0.1,          # Upper bound for vol_median, e.g. 10%
    verbose=True
):
    """
    Logic:
        1. Compute daily returns from training close prices.
           Optionally winsorize both return tails by winsor_pct%.

        2. Estimate rolling volatility with vol_lookback,
           then use the median rolling volatility as vol_median.
           If specified, clamp vol_median into [min_vol, max_vol].

        3. Scale base_th_ref linearly by the ratio between
           vol_median and ref_vol:
               base_th_data = base_th_ref * (vol_median / ref_vol)

        4. Apply floor / ceiling protection:
               base_th_floor = floor_mult * base_th_ref
               base_th_ceil  = ceil_mult  * base_th_ref
               base_th = min(max(base_th_data, base_th_floor), base_th_ceil)

    Parameters:
        train_df     : training DataFrame; must contain close_col
        close_col    : close price column name
        base_th_ref  : reference threshold when daily vol is close to ref_vol
        vol_lookback : rolling volatility window length
        ref_vol      : reference volatility, typically 0.01 for 1%
        floor_mult   : lower-bound multiplier relative to base_th_ref
        ceil_mult    : upper-bound multiplier relative to base_th_ref
        winsor_pct   : two-sided winsorization percentage for daily returns
                       None or 0 disables winsorization
        min_vol      : lower bound for vol_median; None disables lower bound
        max_vol      : upper bound for vol_median; None disables upper bound
        verbose      : whether to print intermediate diagnostics

    Returns:
        base_th (float):
            Volatility-adjusted threshold that can be passed directly
            to cta_backtest_money as base_th.
    """
    # 1) Compute daily returns from training close prices
    if close_col not in train_df.columns:
        raise KeyError(f"[choose_base_th_from_train] '{close_col}' not found in train_df columns.")

    close = train_df[close_col].astype(float)
    ret = close.pct_change().dropna()

    if len(ret) < vol_lookback:
        if verbose:
            print("[choose_base_th_from_train] Too few data points, returning base_th_ref directly.")
        return float(base_th_ref)

    # 1.5) Light winsorization for heavy-tail robustness
    if winsor_pct is not None and winsor_pct > 0:
        # winsor_pct = 1 means clipping at the 1% and 99% quantiles
        low_q, high_q = np.percentile(ret, [winsor_pct, 100 - winsor_pct])
        ret = ret.clip(low_q, high_q)

    # 2) Estimate median rolling volatility on the training set
    rolling_vol = ret.rolling(vol_lookback).std()
    vol_median_raw = rolling_vol.median()

    if (vol_median_raw is None) or (not np.isfinite(vol_median_raw)) or (vol_median_raw <= 0):
        if verbose:
            print("[choose_base_th_from_train] vol_median invalid, returning base_th_ref directly.")
        return float(base_th_ref)

    vol_median = float(vol_median_raw)

    # 2.5) Clamp vol_median into a reasonable range if bounds are specified
    vol_median_before_clip = vol_median

    if min_vol is not None:
        vol_median = max(vol_median, float(min_vol))
    if max_vol is not None:
        vol_median = min(vol_median, float(max_vol))

    # 3) Scale base_th by the training-set volatility level
    base_th_data = base_th_ref * (vol_median / ref_vol)

    # 4) Apply floor / ceiling protection
    base_th_floor = floor_mult * base_th_ref
    base_th_ceil  = ceil_mult  * base_th_ref

    base_th_clipped = max(base_th_data, base_th_floor)
    base_th_clipped = min(base_th_clipped, base_th_ceil)

    if verbose:
        print(
            "[choose_base_th_from_train] "
            f"vol_median_raw={vol_median_before_clip:.4f}, "
            f"vol_median_used={vol_median:.4f}, "
            f"base_th_data={base_th_data:.4f}, "
            f"floor={base_th_floor:.4f}, ceil={base_th_ceil:.4f} "
            f"→ base_th={base_th_clipped:.4f}"
        )

    return float(base_th_clipped)
