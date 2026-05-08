# -*- coding: utf-8 -*-
"""Auto-split from the original single-contract notebooks.
Core formulas, thresholds, and function bodies are kept unchanged.
"""

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, linregress

def infer_factor_sign(
    df,
    factor,
    future_col="future_ret_H",
    n_group=5,
    min_samples=300,
    bootstrap_n=200,
    min_confidence=0.7
):
    """
    Infer the directional sign of a factor.

    Components:
        1. Quantile bucketing with qcut
        2. IC sign as the linear direction proxy
        3. Monotonicity check
        4. Bootstrap direction stability
        5. Final weighted direction decision

    Returns:
        final_sign: +1 / -1 / 0
        detail: dict containing all intermediate diagnostics
    """

    # ======================
    # Prepare clean sample
    # ======================
    sub = df[[factor, future_col]].dropna().copy()
    if len(sub) < min_samples:
        return 0, {"error": "too_few_samples"}

    x = sub[factor].values
    y = sub[future_col].values

    # ======================
    # 1) Quantile bucketing
    # ======================
    try:
        sub["group"] = pd.qcut(x, n_group, labels=False, duplicates="drop")
        g = sub.groupby("group")[future_col].mean()
        qcut_sign = np.sign(g.iloc[-1] - g.iloc[0])
    except Exception:
        qcut_sign = 0
        g = None  

    # ======================
    # 2) IC sign
    # ======================
    IC = np.corrcoef(x, y)[0, 1]
    if np.isnan(IC):
        ic_sign = 0
    else:
        ic_sign = np.sign(IC)

    # ======================
    # 3) Monotonicity check
    # ======================
    try:
        # Use quantile-bucket group means to test monotonicity
        if g is None:
            mono_sign = 0
            spearman_r = np.nan
        else:
            group_returns = g.values
            group_index = np.arange(len(group_returns))
            spearman_r, _ = spearmanr(group_index, group_returns)
            mono_sign = np.sign(spearman_r) if not np.isnan(spearman_r) else 0
    except Exception:
        mono_sign = 0
        spearman_r = np.nan

    # ========================================
    # 3b) Linear-regression slope direction + t-stat
    # ========================================
    
    reg_sign = 0            
    reg_tstat = np.nan
    reg_slope = np.nan

    try:
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        slope, intercept, r_value, p_value, std_err = linregress(x_centered, y_centered)
        reg_slope = slope

        if (std_err is None) or (std_err == 0) or np.isnan(std_err) or np.isnan(slope):
            reg_sign = 0
            reg_tstat = np.nan
        else:
            reg_sign = int(np.sign(slope))
            reg_tstat = slope / std_err
    except Exception:
        reg_sign = 0
        reg_tstat = np.nan
        reg_slope = np.nan

    # ======================
    # 4) Bootstrap stability
    # ======================
    signs = []
    n = len(sub)

    # Block bootstrap:
    # each block is about 5% of the sample, with at least 5 observations
    block_size = max(5, n // 20)
    n_blocks = int(np.ceil(n / block_size))

    for _ in range(bootstrap_n):
        idx_list = []
        for _ in range(n_blocks):
            start_max = n - block_size
            if start_max < 0:
                start_max = 0
            start = np.random.randint(0, start_max + 1)
            idx_block = range(start, min(start + block_size, n))
            idx_list.extend(idx_block)

        idx = np.array(idx_list[:n]) 
        xb = x[idx]
        yb = y[idx]

        # Bootstrap qcut sign using the same logic as the main sample
        try:
            dfb = pd.DataFrame({"f": xb, "r": yb}).dropna()
            dfb["g"] = pd.qcut(
                dfb["f"], n_group,
                labels=False, duplicates="drop"
            )
            gb = dfb.groupby("g")["r"].mean()
            bsign = np.sign(gb.iloc[-1] - gb.iloc[0])
        except Exception:
            bsign = 0

        signs.append(bsign)

    signs = np.array(signs)
    pos_prob = np.mean(signs == 1)
    neg_prob = np.mean(signs == -1)

    # Bootstrap-implied dominant direction
    if pos_prob > neg_prob:
        bootstrap_sign = 1
    elif neg_prob > pos_prob:
        bootstrap_sign = -1
    else:
        bootstrap_sign = 0

    bootstrap_confidence = max(pos_prob, neg_prob)

    # ======================
    # 5) Final weighted direction decision
    # ======================

    # Ensemble weights
    w_qcut = 0.35
    w_ic   = 0.35
    w_mono = 0.2
    w_boot = 0.1

    raw_score = (
        w_qcut * qcut_sign +
        w_ic   * ic_sign +
        w_mono * mono_sign +
        w_boot * bootstrap_sign
    )

    # Regression constraint:
    # t-stat threshold + mild reinforcement weight
    reg_t_min = 2.0   # |t| >= 2 is treated as statistically significant
    w_reg = 0.15      # small boost; should not dominate the ensemble

    # Use regression slope direction as a hard constraint
    # and as a consistency enhancer
    if not np.isnan(reg_tstat) and abs(reg_tstat) >= reg_t_min and reg_sign != 0:
        raw_sign_before_reg = np.sign(raw_score)

        if raw_sign_before_reg != 0:
            if raw_sign_before_reg != reg_sign:
                # Case 1:
                # Significant regression points in the opposite direction.
                # Veto the ensemble direction.
                raw_score = 0.0
            else:
                # Case 2:
                # Regression direction is consistent with the ensemble.
                # Modestly reinforce raw_score.
                raw_score = raw_score + w_reg * reg_sign
        # If raw_score == 0, keep it at 0.
        # Do not let regression alone decide the direction.

    final_sign = int(np.sign(raw_score))

    # Bootstrap confidence filter:
    # if the direction is unstable, drop the factor
    if bootstrap_confidence < min_confidence:
        final_sign = 0

    # ======================
    # Pack and return results
    # ======================
    detail = {
        "qcut_sign": int(qcut_sign),
        "ic": IC,
        "ic_sign": int(ic_sign),
        "spearman": spearman_r,
        "mono_sign": int(mono_sign),
        "bootstrap_sign": int(bootstrap_sign),
        "bootstrap_confidence": bootstrap_confidence,
        "final_sign_rawscore": raw_score,
        "final_sign": int(final_sign),

        # Regression diagnostics
        "reg_slope": float(reg_slope) if not np.isnan(reg_slope) else np.nan,
        "reg_tstat": float(reg_tstat) if not np.isnan(reg_tstat) else np.nan,
        "reg_sign": int(reg_sign),

        # Block-bootstrap configuration
        "bootstrap_block_size": int(block_size),
        "bootstrap_n": int(bootstrap_n),
    }

    return final_sign, detail

def build_sign_dict(
    factor_list,
    train_df,
    future_col="future_ret_H",
    n_group=5,
    bootstrap_n=300,
    min_confidence=0.7
):
    """
    Build {factor -> sign} by running infer_factor_sign
    on the training set.

    Returns
    -------
    sign_dict : dict
        {factor_name: +1 / -1 / 0}
    detail_dict : dict
        {factor_name: full diagnostics returned by infer_factor_sign}
    """
    sign_dict = {}
    detail_dict = {}
    for f in factor_list:
        sign, detail = infer_factor_sign(
            train_df,
            factor=f,
            future_col=future_col,
            n_group=n_group,
            bootstrap_n=bootstrap_n,
            min_confidence=min_confidence
        )
        sign = int(sign)
        sign_dict[f] = sign
        detail_dict[f] = detail
    return sign_dict, detail_dict


up_dict, up_detail = build_sign_dict(up_factors, train_all)
down_dict, down_detail = build_sign_dict(down_factors, train_all)


# 3) Keep only factors with non-zero inferred signs,
#    while preserving the original factor order
