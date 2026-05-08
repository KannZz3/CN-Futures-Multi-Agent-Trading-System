# -*- coding: utf-8 -*-
"""Auto-split from the original single-contract notebooks.
Core formulas, thresholds, and function bodies are kept unchanged.
"""

import numpy as np
import pandas as pd

def load_weekly_factor_lagged(
    factor_path: str,
    date_col: str = "date",
    factor_col: str = None,
    new_col: str = None,
    lag_weeks: int = 1
):
    """
    Load a weekly factor and apply shift(lag_weeks),
    so daily data in the current week can only use
    the previous week's factor value.

    Parameters
    ----------
    factor_path : str
        Path to the weekly factor CSV.
    date_col : str
        Name of the date column.
    factor_col : str
        Original factor column name.
    new_col : str
        New factor column name after attaching to daily data.
    lag_weeks : int
        Number of weeks to lag. Default is 1 to avoid weekly-data leakage.
    """

    if factor_col is None:
        raise ValueError("factor_col must be specified")

    if new_col is None:
        new_col = factor_col

    fac = pd.read_csv(factor_path)
    fac[date_col] = pd.to_datetime(fac[date_col])
    fac = fac.sort_values(date_col).set_index(date_col)

    if factor_col not in fac.columns:
        raise ValueError(f"{factor_path} missing column: {factor_col}")

    out = fac[[factor_col]].copy()
    out = out.rename(columns={factor_col: new_col})

    # Key step:
    # lag the entire weekly factor by one week
    out[new_col] = out[new_col].shift(lag_weeks)

    return out


# ============================================================
# 2. Align weekly factor to daily test data
# ============================================================

def attach_weekly_factor_to_daily(
    daily_df: pd.DataFrame,
    weekly_factor_lagged: pd.DataFrame,
    factor_col: str,
    week_freq: str = "W-FRI"
):
    """
    Align an already-lagged weekly factor to daily data.

    Logic:
    - Map each daily date to its corresponding weekly Friday week_end.
    - Use week_end to match the lagged weekly factor.
    - Therefore, all trading days in the current week only receive
      the previous week's factor value.
    """

    if not isinstance(daily_df.index, pd.DatetimeIndex):
        raise ValueError("daily_df index must be a DatetimeIndex")

    daily = daily_df.copy().sort_index()

    if factor_col in daily.columns:
        daily = daily.drop(columns=[factor_col])

    fac = weekly_factor_lagged.copy().sort_index()

    if factor_col not in fac.columns:
        raise ValueError(f"weekly_factor_lagged missing column: {factor_col}")

    # Map each daily date to the Friday of its week
    daily_week_end = daily.index.to_period(week_freq).to_timestamp(week_freq)

    factor_map = fac[factor_col]

    daily[factor_col] = daily_week_end.map(factor_map)

    return daily


# ============================================================
# 3. Attach RM warehouse factor and RM basis factor
# ============================================================

def attach_rm_wh_and_basis_factors(
    daily_df: pd.DataFrame,
    wh_factor_path: str = "RM_warehouse_factor_with_wh_factor.csv",
    basis_factor_path: str = "RM_weekly_dom_basis_with_basis_factor_standardized.csv",
    wh_source_col: str = "wh_factor",
    basis_source_col: str = "basis_factor_zscore_近三周mean",
    wh_new_col: str = "wh_factor",
    basis_new_col: str = "basis_factor",
    week_freq: str = "W-FRI",
    lag_weeks: int = 1
):
    """
    Attach two weekly factors to daily test data:
    1. Warehouse factor: wh_factor
    2. Basis factor: basis_factor

    Both factors use previous-week information only
    to avoid time leakage.
    """

    daily = daily_df.copy().sort_index()

    # ---------- Warehouse factor ----------
    wh_lagged = load_weekly_factor_lagged(
        factor_path=wh_factor_path,
        date_col="date",
        factor_col=wh_source_col,
        new_col=wh_new_col,
        lag_weeks=lag_weeks
    )

    daily = attach_weekly_factor_to_daily(
        daily_df=daily,
        weekly_factor_lagged=wh_lagged,
        factor_col=wh_new_col,
        week_freq=week_freq
    )

    # ---------- Basis factor ----------
    basis_lagged = load_weekly_factor_lagged(
        factor_path=basis_factor_path,
        date_col="date",
        factor_col=basis_source_col,
        new_col=basis_new_col,
        lag_weeks=lag_weeks
    )

    daily = attach_weekly_factor_to_daily(
        daily_df=daily,
        weekly_factor_lagged=basis_lagged,
        factor_col=basis_new_col,
        week_freq=week_freq
    )

    return daily, wh_lagged, basis_lagged


# ============================================================
# 4. Run factor attachment
# ============================================================

def check_first_test_week_factor_alignment(
    daily_df: pd.DataFrame,
    wh_lagged: pd.DataFrame,
    basis_lagged: pd.DataFrame,
    wh_col: str = "wh_factor",
    basis_col: str = "basis_factor",
    week_freq: str = "W-FRI",
    n_days: int = 5
):
    """
    Check the first n_days trading days after the test period starts.

    The goal is to verify whether those daily rows all use
    the previous week's warehouse factor and basis factor.
    """

    if not isinstance(daily_df.index, pd.DatetimeIndex):
        raise ValueError("daily_df.index must be a DatetimeIndex")

    daily = daily_df.copy().sort_index()
    wh = wh_lagged.copy().sort_index()
    basis = basis_lagged.copy().sort_index()

    first_days = daily.index[:n_days]

    rows = []

    for dt in first_days:
        current_week_end = dt.to_period(week_freq).to_timestamp(week_freq)
        source_week_used = current_week_end - pd.Timedelta(days=7)

        wh_value = wh.loc[current_week_end, wh_col] if current_week_end in wh.index else np.nan
        basis_value = basis.loc[current_week_end, basis_col] if current_week_end in basis.index else np.nan

        rows.append({
            "daily_date": dt,
            "current_week_end": current_week_end,
            "expected_source_week": source_week_used,
            "mapped_wh_factor": wh_value,
            "mapped_basis_factor": basis_value,
            "daily_wh_factor": daily.loc[dt, wh_col] if wh_col in daily.columns else np.nan,
            "daily_basis_factor": daily.loc[dt, basis_col] if basis_col in daily.columns else np.nan,
        })

    check_df = pd.DataFrame(rows)

    print("=" * 100)
    print("First test week factor alignment check")
    print("=" * 100)
    print(check_df)

    print("\nCheck whether all first days use the same lagged weekly factor:")
    print("unique wh_factor:", check_df["daily_wh_factor"].nunique(dropna=False))
    print("unique basis_factor:", check_df["daily_basis_factor"].nunique(dropna=False))

    return check_df
