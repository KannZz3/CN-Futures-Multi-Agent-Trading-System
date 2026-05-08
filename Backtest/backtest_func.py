# -*- coding: utf-8 -*-
"""
Modularized from:
- 单合约全流程_技术+周线.ipynb
- 单合约全流程_技术+基本面.ipynb

Design rule:
- Keep the original single-contract logic as much as possible.
- Use train-fitted statistics for score / thresholds.
- Use one-week lag for weekly filters and fundamental factors to avoid leakage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates


def backtest_money(
    df,
    long_score_col="score_long",
    short_score_col="score_short",
    T_long=None,
    T_short=None,
    base_th=0.01,
    max_hold_days=10,
    init_capital=500000,
    pos_pct=0.3,
    leverage=10,
    shift_signal=True,

    # ====== Weekly filter parameters ======
    use_weekly_filter=True,
    wk_trend_col="wk_trend_sign",
    wk_vol_regime_col="wk_vol_regime",
    wk_crowded_long_col="wk_crowded_long",
    block_vol_regime=2,
    allow_trend_zero_open=False,

    # ====== Warehouse filter parameters ======
    use_wh_filter=True,
    wh_score_col="wh_factor",
    wh_long_block_th=-0.2,
    wh_short_block_th=0.2,

    # ====== Basis filter parameters ======
    use_basis_filter=True,
    basis_score_col="basis_factor",
    basis_long_block_th=-0.2,
    basis_short_block_th=0.2,

    # ====== Performance statistics parameters ======
    rf_annual=0.012,
    annual_trading_days=252
):
    df = df.copy()

    # === 0) Compute underlying daily returns ===
    df["ret"] = df["close"].pct_change().fillna(0.0)

    # === 0.5) Validate thresholds and required columns ===
    if long_score_col not in df.columns:
        raise KeyError(f"[backtest_money] '{long_score_col}' not found in df columns.")

    if short_score_col not in df.columns:
        raise KeyError(f"[backtest_money] '{short_score_col}' not found in df columns.")

    if T_long is None or T_short is None:
        raise ValueError("[backtest_money] Please provide both T_long and T_short.")

    T_long = float(T_long)
    T_short = float(T_short)

    if T_long <= 0:
        raise ValueError("[backtest_money] T_long should be positive.")

    if T_short <= 0:
        raise ValueError("[backtest_money] T_short should be positive in split-score mode.")

    # === 1) Construct raw directional signals from split scores ===
    long_score = df[long_score_col].astype(float)
    short_score = df[short_score_col].astype(float)

    long_signal = long_score > T_long
    short_signal = short_score > T_short

    long_strength = long_score / T_long
    short_strength = short_score / T_short

    sig = pd.Series(0.0, index=df.index, dtype=float)

    sig[long_signal & ~short_signal] = 1.0
    sig[short_signal & ~long_signal] = -1.0

    both_signal = long_signal & short_signal
    sig[both_signal & (long_strength >= short_strength)] = 1.0
    sig[both_signal & (short_strength > long_strength)] = -1.0

    if shift_signal:
        sig = sig.shift(1).fillna(0.0)

    df["signal_raw"] = sig
    df["long_signal_raw"] = long_signal.astype(float)
    df["short_signal_raw"] = short_signal.astype(float)
    df["long_strength"] = long_strength
    df["short_strength"] = short_strength

    # === 1.5) Apply weekly filters ===
    weekly_ok = pd.Series(True, index=df.index, dtype=bool)

    if use_weekly_filter:
        required_wk_cols = [
            wk_trend_col,
            wk_vol_regime_col,
            wk_crowded_long_col,
        ]

        missing_wk_cols = [c for c in required_wk_cols if c not in df.columns]

        if missing_wk_cols:
            raise ValueError(
                f"use_weekly_filter=True, but missing weekly columns: {missing_wk_cols}"
            )

        trend = df[wk_trend_col]
        vol_reg = df[wk_vol_regime_col]
        crowded = df[wk_crowded_long_col]

        weekly_ok &= trend.notna()
        weekly_ok &= vol_reg.notna()
        weekly_ok &= crowded.notna()

        if not allow_trend_zero_open:
            weekly_ok &= (trend != 0)

        same_dir_or_zero = ((trend * sig) >= 0) | (sig == 0)
        weekly_ok &= same_dir_or_zero.fillna(False)

        high_vol_mask = (vol_reg == block_vol_regime)
        weekly_ok &= ~(high_vol_mask & (sig != 0))

        crowded = crowded.fillna(0.0)
        weekly_block_long = (crowded >= 0.5) & (sig > 0)
        weekly_ok &= ~weekly_block_long

    # === 1.6) Apply warehouse filter ===
    wh_ok = pd.Series(True, index=df.index, dtype=bool)

    df["wh_block_long"] = 0.0
    df["wh_block_short"] = 0.0

    if use_wh_filter:
        if wh_score_col not in df.columns:
            raise ValueError(
                f"use_wh_filter=True, but '{wh_score_col}' not found in df columns."
            )

        wh_score = df[wh_score_col].astype(float)

        wh_block_long = (sig > 0) & (wh_score < wh_long_block_th)
        wh_block_short = (sig < 0) & (wh_score > wh_short_block_th)

        df["wh_block_long"] = wh_block_long.astype(float)
        df["wh_block_short"] = wh_block_short.astype(float)

        wh_ok &= wh_score.notna()
        wh_ok &= ~(wh_block_long | wh_block_short)

    # === 1.7) Apply basis filter ===
    basis_ok = pd.Series(True, index=df.index, dtype=bool)

    df["basis_block_long"] = 0.0
    df["basis_block_short"] = 0.0

    if use_basis_filter:
        if basis_score_col not in df.columns:
            raise ValueError(
                f"use_basis_filter=True, but '{basis_score_col}' not found in df columns."
            )

        basis_score = df[basis_score_col].astype(float)

        basis_block_long = (sig > 0) & (basis_score < basis_long_block_th)
        basis_block_short = (sig < 0) & (basis_score > basis_short_block_th)

        df["basis_block_long"] = basis_block_long.astype(float)
        df["basis_block_short"] = basis_block_short.astype(float)

        basis_ok &= basis_score.notna()
        basis_ok &= ~(basis_block_long | basis_block_short)

    # === 1.8) Final open-position filter ===
    fund_ok = wh_ok & basis_ok
    open_ok = weekly_ok & fund_ok

    sig_eff = sig.copy()
    sig_eff[~open_ok] = 0.0

    df["signal_eff"] = sig_eff
    df["weekly_ok"] = weekly_ok.astype(float)
    df["wh_ok"] = wh_ok.astype(float)
    df["basis_ok"] = basis_ok.astype(float)
    df["fund_ok"] = fund_ok.astype(float)
    df["open_ok"] = open_ok.astype(float)

    # === 2) Initialize strategy state ===
    capital = init_capital
    pos_dir = 0.0
    hold_value = 0.0
    cum_ret = 0.0
    hold_days = 0
    reached_tp1 = False

    capital_list = []
    pos_exposure_list = []
    hold_value_list = []

    trade_returns = []
    exit_reasons = []

    # === 3) Main backtest loop ===
    for i in range(len(df)):
        daily_ret_raw = df["ret"].iloc[i]
        signal_today = df["signal_eff"].iloc[i]

        # A. Settle existing position first
        if pos_dir != 0.0:
            effective_notional = pos_dir * leverage * hold_value
            daily_profit = effective_notional * daily_ret_raw
            capital += daily_profit

            cum_ret += pos_dir * leverage * daily_ret_raw
            hold_days += 1

            if cum_ret >= base_th:
                reached_tp1 = True

            if capital <= 0:
                trade_returns.append(cum_ret)
                exit_reasons.append("liquidation")

                capital = 0.0
                pos_dir = 0.0
                hold_value = 0.0
                cum_ret = 0.0
                hold_days = 0
                reached_tp1 = False

                capital_list.append(capital)
                pos_exposure_list.append(0.0)
                hold_value_list.append(0.0)

                for _ in range(len(df) - i - 1):
                    capital_list.append(0.0)
                    pos_exposure_list.append(0.0)
                    hold_value_list.append(0.0)

                break

            # B. Exit rules
            exit_flag = False
            exit_reason = None

            if cum_ret >= 2.5 * base_th:
                exit_flag = True
                exit_reason = "take_profit"
            elif cum_ret <= -0.9 * base_th:
                exit_flag = True
                exit_reason = "stop_loss"
            elif reached_tp1 and cum_ret <= 0.7 * base_th:
                exit_flag = True
                exit_reason = "trailing_stop"
            elif hold_days >= max_hold_days:
                exit_flag = True
                exit_reason = "max_hold"

            if exit_flag:
                trade_returns.append(cum_ret)
                exit_reasons.append(exit_reason)

                pos_dir = 0.0
                hold_value = 0.0
                cum_ret = 0.0
                hold_days = 0
                reached_tp1 = False

        # C. Open new position after settlement / exit
        if pos_dir == 0.0 and signal_today != 0.0 and capital > 0:
            pos_dir = signal_today
            hold_value = capital * pos_pct
            cum_ret = 0.0
            hold_days = 0
            reached_tp1 = False

        # D. Record daily portfolio state
        capital_list.append(capital)

        pos_exposure = (
            pos_dir * leverage * hold_value / capital
            if capital > 0 else 0.0
        )

        pos_exposure_list.append(pos_exposure)
        hold_value_list.append(hold_value)

    # === 4) Write back result columns ===
    df["capital"] = capital_list
    df["pos"] = pos_exposure_list
    df["hold_value"] = hold_value_list
    df["equity"] = df["capital"] / float(init_capital)

    if pos_dir != 0.0:
        trade_returns.append(cum_ret)
        exit_reasons.append("still_open_at_end")

    # === 5) Compute performance statistics ===
    equity_series = pd.Series(capital_list, index=df.index)
    port_ret = equity_series.pct_change().fillna(0.0)

    rf_daily = (1.0 + rf_annual) ** (1.0 / annual_trading_days) - 1.0
    excess_ret = port_ret - rf_daily

    if excess_ret.std() > 0:
        sharpe_annual = (
            excess_ret.mean() / excess_ret.std()
        ) * np.sqrt(annual_trading_days)
    else:
        sharpe_annual = np.nan

    trade_returns = np.asarray(trade_returns, dtype=float)
    n_trades = int(trade_returns.size)
    n_win = int((trade_returns > 0).sum())
    n_loss = int((trade_returns < 0).sum())
    win_rate = float(n_win / n_trades) if n_trades > 0 else np.nan

    if n_loss > 0:
        worst_losses = np.sort(trade_returns[trade_returns < 0])
        worst5_loss_ret = worst_losses[:5].tolist()
    else:
        worst5_loss_ret = []

    exit_reason_counts = pd.Series(exit_reasons).value_counts().to_dict()

    # === 6) Diagnostic statistics ===
    active_signal_mask = df["signal_raw"] != 0

    wh_block_any = (df["wh_block_long"] == 1) | (df["wh_block_short"] == 1)
    basis_block_any = (df["basis_block_long"] == 1) | (df["basis_block_short"] == 1)

    bt_stats = {
        "n_trades": n_trades,
        "n_win": n_win,
        "n_loss": n_loss,
        "win_rate": win_rate,
        "worst5_loss_ret": worst5_loss_ret,
        "sharpe_annual": sharpe_annual,
        "rf_annual": rf_annual,
        "annual_trading_days": annual_trading_days,
        "T_long": T_long,
        "T_short": T_short,
        "base_th": base_th,
        "long_score_col": long_score_col,
        "short_score_col": short_score_col,

        "use_weekly_filter": use_weekly_filter,
        "use_wh_filter": use_wh_filter,
        "use_basis_filter": use_basis_filter,
        "wh_score_col": wh_score_col,
        "basis_score_col": basis_score_col,
        "wh_long_block_th": wh_long_block_th,
        "wh_short_block_th": wh_short_block_th,
        "basis_long_block_th": basis_long_block_th,
        "basis_short_block_th": basis_short_block_th,

        # Diagnostic counts
        "raw_signal_count": int(active_signal_mask.sum()),
        "effective_signal_count": int((df["signal_eff"] != 0).sum()),

        "wh_block_long_count": int(df["wh_block_long"].sum()),
        "wh_block_short_count": int(df["wh_block_short"].sum()),
        "wh_block_any_count": int(wh_block_any.sum()),

        "basis_block_long_count": int(df["basis_block_long"].sum()),
        "basis_block_short_count": int(df["basis_block_short"].sum()),
        "basis_block_any_count": int(basis_block_any.sum()),

        "fund_block_count": int((active_signal_mask & (~fund_ok)).sum()),
        "open_block_count": int((active_signal_mask & (~open_ok)).sum()),

        "exit_reason_counts": exit_reason_counts,
        "exit_reasons": exit_reasons,
    }

    df.attrs["bt_stats"] = bt_stats

    return df



def plot_backtest(
    bt_df,
    contract_name="Contract",
    save_png=False,
    output_path=None
):
    stats = bt_df.attrs.get("bt_stats", {})

    n_trades   = stats.get("n_trades", np.nan)
    n_win      = stats.get("n_win", np.nan)
    n_loss     = stats.get("n_loss", np.nan)
    win_rate   = stats.get("win_rate", np.nan)
    worst5_ret = stats.get("worst5_loss_ret", [])
    sharpe     = stats.get("sharpe_annual", np.nan)
    rf_annual  = stats.get("rf_annual", 0.012)

    use_basis_filter = stats.get("use_basis_filter", np.nan)
    basis_long_th    = stats.get("basis_long_block_th", np.nan)
    basis_short_th   = stats.get("basis_short_block_th", np.nan)

    use_wh_filter = stats.get("use_wh_filter", np.nan)
    wh_long_th    = stats.get("wh_long_block_th", np.nan)
    wh_short_th   = stats.get("wh_short_block_th", np.nan)

    win_rate_str = "N/A" if pd.isna(win_rate) else f"{win_rate * 100:.2f}%"
    sharpe_str   = "N/A" if pd.isna(sharpe) else f"{sharpe:.3f}"
    rf_str       = f"{rf_annual * 100:.2f}%"

    worst5_str = (
        ", ".join([f"{x * 100:.1f}%" for x in worst5_ret])
        if len(worst5_ret) > 0 else "N/A"
    )

    def fmt_th(x):
        return "N/A" if pd.isna(x) else f"{x:.2f}"

    ret = (bt_df["equity"] - 1).dropna()

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    fig.subplots_adjust(right=0.74)

    ax.plot(ret.index, ret, label=contract_name, linewidth=1.5)

    ax.set_title(f"{contract_name} Cumulative Return", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=30)

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-1.0, color="red", linestyle=":", linewidth=1.0, alpha=0.85)

    ax.text(
        ret.index[0],
        -1.0,
        "  -100% capital wipeout",
        color="red",
        fontsize=8,
        va="bottom",
        ha="left"
    )

    y_min = min(ret.min(), -1.0)
    y_max = ret.max()
    ax.set_ylim(y_min - 0.05, y_max + 0.10)

    ax.grid(True, linestyle="--", alpha=0.35)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper left", frameon=False, title="Contract")

    # ===== Right-side summary =====
    x_stats = 0.77
    y_top   = 0.87
    line_h  = 0.047

    fig.text(x_stats, y_top, "Backtest Summary",
             ha="left", va="top", fontsize=11, fontweight="bold")

    fig.text(x_stats, y_top - 1.0 * line_h,
             f"Total trades: {n_trades}", fontsize=9)

    fig.text(x_stats, y_top - 2.0 * line_h,
             f"Winning trades: {n_win}", fontsize=9)

    fig.text(x_stats, y_top - 3.0 * line_h,
             f"Losing trades: {n_loss}", fontsize=9)

    fig.text(x_stats, y_top - 4.0 * line_h,
             f"Hit ratio: {win_rate_str}", fontsize=9)

    fig.text(x_stats, y_top - 5.2 * line_h,
             "Worst 5 trades (strategy R):", fontsize=9)

    fig.text(x_stats, y_top - 6.2 * line_h,
             worst5_str, fontsize=9)

    fig.text(x_stats, y_top - 7.4 * line_h,
             f"Annual Sharpe (rf={rf_str}): {sharpe_str}", fontsize=9)

    fig.text(x_stats, y_top - 8.7 * line_h,
             "Filter Settings", fontsize=9, fontweight="bold")

    fig.text(x_stats, y_top - 9.7 * line_h,
             f"Basis filter: {use_basis_filter}", fontsize=8)

    fig.text(x_stats, y_top - 10.6 * line_h,
             f"Basis th: L={fmt_th(basis_long_th)}, S={fmt_th(basis_short_th)}",
             fontsize=8)

    fig.text(x_stats, y_top - 11.6 * line_h,
             f"WH filter: {use_wh_filter}", fontsize=8)

    fig.text(x_stats, y_top - 12.5 * line_h,
             f"WH th: L={fmt_th(wh_long_th)}, S={fmt_th(wh_short_th)}",
             fontsize=8)

    fig.text(x_stats, y_top - 13.8 * line_h,
             "Note: -100% means capital = 0",
             fontsize=8, color="red")

    if save_png:
        if output_path is None:
            output_path = f"{contract_name}_xgb_f.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")

    plt.show()