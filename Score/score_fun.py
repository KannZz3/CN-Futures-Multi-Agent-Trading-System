# -*- coding: utf-8 -*-
"""Auto-split from the original single-contract notebooks.
Core formulas, thresholds, and function bodies are kept unchanged.
"""

import numpy as np
import pandas as pd

def add_score(
    df,
    up_factors,
    down_factors,
    up_sign_dict,
    down_sign_dict,
    lookback=60,
    winsor_limit=5,
    use_robust=True,
    remove_corr=True,
    corr_threshold=0.6,
    factor_group_map=None,
    conflict_check=True,   
    conflict_threshold=0.4
):
    """
    Split-score version of add_score.

    Core logic:
        - up_factors are used only for score_long
        - down_factors are used only for score_short
        - No long-short combined score is constructed
        - No 3-day smoothing is applied
        - Correlation filtering is performed separately
          for up/down factor groups
        - Conflict suppression is disabled by default

    Outputs:
        score_long  :
            Long-side signal strength;
            larger values indicate stronger bullish signals

        score_short :
            Short-side signal strength;
            larger values indicate stronger bearish signals
    """

    df = df.copy()

    # =============================
    # 0. Clean factor lists and
    #    filter conflicting signs
    # =============================
    up_factors = list(dict.fromkeys(up_factors))
    down_factors = list(dict.fromkeys(down_factors))

    overlap = set(up_factors) & set(down_factors)
    conflict_drop = set()

    for f in overlap:
        su = np.sign(up_sign_dict.get(f, 0.0))
        sd = np.sign(down_sign_dict.get(f, 0.0))

        if su != 0 and sd != 0 and su != sd:
            print(
                f"[Warning] Factor {f} has opposite directions "
                f"in up_sign_dict / down_sign_dict. "
                f"Removing from both long/short factor lists."
            )
            conflict_drop.add(f)

    if conflict_drop:
        up_factors = [f for f in up_factors if f not in conflict_drop]
        down_factors = [f for f in down_factors if f not in conflict_drop]

    # =============================
    # 1. Remove highly correlated
    #    factors separately for
    #    up/down groups
    # =============================
    def _remove_corr_within_group(factors, df, corr_threshold):
        factors = [f for f in factors if f in df.columns]

        if len(factors) <= 1:
            return factors

        corr = df[factors].corr().abs()
        keep = []

        for f in factors:
            if not keep:
                keep.append(f)
                continue

            if all(corr.loc[f, k] < corr_threshold for k in keep):
                keep.append(f)

        return keep

    if remove_corr:
        up_factors = _remove_corr_within_group(
            up_factors, df, corr_threshold
        )

        down_factors = _remove_corr_within_group(
            down_factors, df, corr_threshold
        )

    all_factors = list(dict.fromkeys(up_factors + down_factors))

    if len(all_factors) == 0:
        df["score_long"] = np.nan
        df["score_short"] = np.nan
        return df

    # =============================
    # 2. Compute rolling z-scores
    # =============================
    valid_factors = []

    for f in all_factors:

        if f not in df.columns:
            print(f"[Skip factor] {f}: not found in df columns")
            continue

        series = df[f].astype(float)

        if use_robust:
            rolling_med = series.rolling(lookback).median()

            rolling_mad = (
                (series - rolling_med)
                .abs()
                .rolling(lookback)
                .median()
            )

            z = (
                (series - rolling_med)
                / (rolling_mad * 1.4826 + 1e-6)
            )

        else:
            rolling_mean = series.rolling(lookback).mean()
            rolling_std = series.rolling(lookback).std()

            z = (
                (series - rolling_mean)
                / (rolling_std + 1e-6)
            )

        if z.isna().all() or z.std(skipna=True) == 0:
            print(
                f"[Skip factor] {f}: invalid z-score "
                f"(all NaN or constant values)"
            )
            continue

        z = z.clip(-winsor_limit, winsor_limit)

        df[f"_z_{f}"] = z.astype(float)

        valid_factors.append(f)

    if len(valid_factors) == 0:
        df["score_long"] = np.nan
        df["score_short"] = np.nan
        return df

    up_valid = [f for f in up_factors if f in valid_factors]
    down_valid = [f for f in down_factors if f in valid_factors]

    valid_union = list(dict.fromkeys(up_valid + down_valid))

    if len(valid_union) == 0:
        df["score_long"] = np.nan
        df["score_short"] = np.nan
        return df

    # =============================
    # 3. Build z-score matrix
    # =============================
    z_cols = [f"_z_{f}" for f in valid_union]

    Z = df[z_cols].copy()
    Z.columns = valid_union

    # Strict validity check:
    # score_long / score_short become valid only
    # when all effective up/down factors have z-scores
    valid_row_mask = Z.notna().all(axis=1)

    up_signs = pd.Series(
        {f: float(np.sign(up_sign_dict.get(f, 0.0))) for f in up_valid},
        dtype=float
    )

    down_signs = pd.Series(
        {f: float(np.sign(down_sign_dict.get(f, 0.0))) for f in down_valid},
        dtype=float
    )

    # =============================
    # 4. Construct long/short scores
    # =============================

    # Long score:
    # constructed only from up_factors
    if len(up_valid) > 0:
        Z_up = Z[up_valid]

        contrib_up = Z_up.mul(up_signs, axis=1)

        # Stronger positive contributions
        # imply stronger bullish signals
        score_long_raw = (
            contrib_up
            .clip(lower=0.0)
            .mean(axis=1)
        )

    else:
        score_long_raw = pd.Series(0.0, index=df.index)

    # Short score:
    # constructed only from down_factors
    if len(down_valid) > 0:
        Z_down = Z[down_valid]

        contrib_down = Z_down.mul(down_signs, axis=1)

        # down_sign still represents the direction
        # relative to future_ret_H.
        # Stronger negative contributions imply
        # stronger bearish signals.
        score_short_raw = (
            (-contrib_down)
            .clip(lower=0.0)
            .mean(axis=1)
        )

    else:
        score_short_raw = pd.Series(0.0, index=df.index)

    # =============================
    # 5. Conflict suppression
    #    (disabled by default)
    # =============================
    if conflict_check:

        long_active = score_long_raw > conflict_threshold
        short_active = score_short_raw > conflict_threshold

        conflict_mask = long_active & short_active

        score_long_raw[conflict_mask] = 0.0
        score_short_raw[conflict_mask] = 0.0

    # =============================
    # 6. Output scores:
    #    no smoothing,
    #    no combined score
    # =============================
    df["score_long"] = score_long_raw
    df["score_short"] = score_short_raw

    # Strict validity enforcement:
    # scores start only after all effective
    # up/down factor z-scores become available
    df.loc[
        ~valid_row_mask,
        ["score_long", "score_short"]
    ] = np.nan

    return df

def choose_T_long_short_from_train(
    train_df,
    long_col="score_long",
    short_col="score_short",
    q_long=0.90,
    q_short=0.90,
    min_obs=20,
    verbose=True
):
    """
    Split-score threshold selection.

    Logic:
        Higher score_long means a stronger long signal.
        Higher score_short means a stronger short signal.

    Therefore:
        T_long  = right-tail quantile of score_long
        T_short = right-tail quantile of score_short

    Note:
        q_short is no longer 0.10; it is also a right-tail quantile.
    """

    if long_col not in train_df.columns:
        raise KeyError(f"'{long_col}' not found in train_df columns.")

    if short_col not in train_df.columns:
        raise KeyError(f"'{short_col}' not found in train_df columns.")

    s_long = (
        train_df[long_col]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    s_short = (
        train_df[short_col]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    # Use only positive effective signals to estimate thresholds
    s_long_pos = s_long[s_long > 0]
    s_short_pos = s_short[s_short > 0]

    if len(s_long_pos) < min_obs:
        raise ValueError(
            f"Too few positive long scores: {len(s_long_pos)} < {min_obs}"
        )

    if len(s_short_pos) < min_obs:
        raise ValueError(
            f"Too few positive short scores: {len(s_short_pos)} < {min_obs}"
        )

    T_long = s_long_pos.quantile(q_long)
    T_short = s_short_pos.quantile(q_short)

    if verbose:
        print(f"[choose_T_split] score_long positive count  = {len(s_long_pos)}")
        print(f"[choose_T_split] score_short positive count = {len(s_short_pos)}")
        print(f"[choose_T_split] T_long  = {long_col}  q={q_long:.2f}: {T_long:.6f}")
        print(f"[choose_T_split] T_short = {short_col} q={q_short:.2f}: {T_short:.6f}")

    return float(T_long), float(T_short)
