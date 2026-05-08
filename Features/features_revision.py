# -*- coding: utf-8 -*-
"""Auto-split from the original single-contract notebooks.
Core formulas, thresholds, and function bodies are kept unchanged.
"""

import numpy as np
import pandas as pd

def check_missing_values(
    data: pd.DataFrame,
    valid_by_finite: bool = False
) -> pd.DataFrame:
    """
    Iterate through all columns (factors) and summarize
    NaN / inf (+inf / -inf) statistics.

    The function reports:
      - Total NaN / total inf
      - NaN / inf inside the valid range
      - NaN / inf outside the valid range

    Parameters
    ----------
    valid_by_finite : bool
        False:
            Valid-range boundaries follow the original logic
            using first / last non-NaN observations.

        True:
            Valid-range boundaries use first / last finite values
            (neither NaN nor inf).
    """
    missing_summary = []

    for col in data.columns:
        s = data[col]

        is_nan = s.isna()
        is_inf = np.isinf(s.to_numpy(dtype=float, copy=False))

        # Used only when valid_by_finite=True
        is_nonfinite = is_nan.to_numpy() | is_inf

        total_nan = int(is_nan.sum())
        total_inf = int(is_inf.sum())

        # =========================
        # Determine valid-range boundaries
        # =========================
        if valid_by_finite:
            finite_mask = ~is_nonfinite

            if finite_mask.any():
                finite_arr = finite_mask.to_numpy()

                first_pos = int(np.argmax(finite_arr))
                last_pos = int(np.where(finite_arr)[0][-1])

                first_valid_index = s.index[first_pos]
                last_valid_index = s.index[last_pos]

            else:
                first_valid_index = None
                last_valid_index = None

        else:
            first_valid_index = s.first_valid_index()
            last_valid_index = s.last_valid_index()

        # =========================
        # Statistics inside / outside valid range
        # =========================
        if (
            first_valid_index is not None
        ) and (
            last_valid_index is not None
        ):
            in_range = s.loc[first_valid_index:last_valid_index]

            in_nan = int(in_range.isna().sum())

            in_inf = int(
                np.isinf(
                    in_range.to_numpy(dtype=float, copy=False)
                ).sum()
            )

            out_nan = total_nan - in_nan
            out_inf = total_inf - in_inf

        else:
            in_nan = in_inf = 0

            out_nan = total_nan
            out_inf = total_inf

        missing_summary.append(
            {
                "Factor": col,

                "Total NaN": total_nan,
                "Total Inf": total_inf,

                "NaN in Valid Range": in_nan,
                "Inf in Valid Range": in_inf,

                "NaN outside Valid Range": out_nan,
                "Inf outside Valid Range": out_inf,

                "First Valid Index": first_valid_index,
                "Last Valid Index": last_valid_index,
            }
        )

    missing_summary_df = pd.DataFrame(missing_summary)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    return missing_summary_df


# =========================
# Check missing-value statistics
# =========================

def fill_missing_values(data: pd.DataFrame):
    data = data.replace([np.inf, -np.inf], np.nan)

    # Get the valid range for each factor
    def get_valid_range(col):
        first_valid_index = data[col].first_valid_index()
        last_valid_index = data[col].last_valid_index()
        if pd.notna(first_valid_index) and pd.notna(last_valid_index):
            return data.loc[first_valid_index:last_valid_index, col]
        else:
            return data[col]  # If no valid range exists, return the full column

    # ========== 4. Special handling after inspection ==========

    # 1) For these slope-type factors, use previous values to fill inf-derived NaN
    special_prev2_cols = [
        "vol_2_slope", "vol_2_slope_3", "vol_2_slope_5", "vol_2_slope_10", "vol_2_slope_25",
        "mom_16_slope", "mom_16_slope_3", "mom_16_slope_5", "mom_16_slope_10", "mom_16_slope_25",
        "mom_8_slope", "mom_8_slope_3", "mom_8_slope_5", "mom_8_slope_10", "mom_8_slope_25",
        "mom_4_slope", "mom_4_slope_3", "mom_4_slope_5", "mom_4_slope_10", "mom_4_slope_25",
        "mom_2_slope", "mom_2_slope_3", "mom_2_slope_5", "mom_2_slope_10", "mom_2_slope_25",
    ]

    for c in special_prev2_cols:
        if c not in data.columns:
            continue
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before ffill {c}, NaN count: {data[c].isna().sum()}")
            data.loc[valid_range.index, c] = data.loc[valid_range.index, c].ffill()
            print(f"After  ffill {c}, NaN count: {data[c].isna().sum()}")

    # 2) For `mom_2_slope_3` and `mom_4_slope_3`, fill NaN with previous values
    for c in ["mom_2_slope_3", "mom_4_slope_3"]:
        if c not in data.columns:
            continue
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
            data[c] = data[c].fillna(method="ffill")
            print(f"After  filling {c}, NaN count: {data[c].isna().sum()}")

    # 3) For ratio-type factors, fill NaN with previous values
    for c in ["body_pct", "upper_shadow_pct", "lower_shadow_pct"]:
        if c not in data.columns:
            continue
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
            data[c] = data[c].fillna(method="ffill")
            print(f"After  filling {c}, NaN count: {data[c].isna().sum()}")

    # 4) Drop rows before the latest first valid value across all factors
    #    This ensures that all factors have valid observations after trimming
    first_pos = []
    for c in data.columns:
        m = data[c].notna().to_numpy()
        if m.any():
            first_pos.append(int(m.argmax()))
    max_leading = max(first_pos) if first_pos else 0

    return data.iloc[max_leading:].copy()


# Assume `data` has already been loaded as a valid DataFrame

class FeaturePreprocessor:
    def __init__(
        self,
        heavy_kurtosis=6,
        clip_pct=(1, 99),
        log_candidates=("volume", "hold"),
        sigma_clip=6,
        robust=False,
        check_finite=True,
    ):
        self.heavy_kurtosis = heavy_kurtosis
        self.clip_low, self.clip_high = clip_pct
        self.log_candidates = tuple(log_candidates)
        self.sigma_clip = sigma_clip
        self.use_robust = robust
        self.check_finite = check_finite

        self.feature_cols = None
        self.heavy_cols = []

        # Percentile clipping bounds for heavy-tail columns:
        # {column: (lower_bound, upper_bound)}
        self.clip_bounds = {}

        # Sigma-clipping statistics for all columns:
        # {column: (mean, std)}
        self.sigma_bounds = {}

        self.scaler = None
        self.is_fitted = False

    # ============================
    # Utility functions
    # ============================
    def _check_no_nan_inf(self, X: pd.DataFrame, where: str):
        if not self.check_finite:
            return

        arr = X.to_numpy(dtype=float, copy=False)

        if np.isfinite(arr).all():
            return

        bad = ~np.isfinite(arr)

        i, j = np.argwhere(bad)[0]

        col = X.columns[j]
        idx = X.index[i]
        v = X.iloc[i, j]

        raise ValueError(
            f"[{where}] found non-finite value at index={idx}, col='{col}', value={v}. "
            "This preprocessor does NOT handle NaN/inf; please clean data before calling."
        )

    # ============================
    # 1) Detect heavy-tail columns
    # ============================
    def _detect_heavy(self, X: pd.DataFrame):
        heavy = []

        for c in self.feature_cols:

            # Skip return / slope-style features
            if ("slope" in c) or ("ret" in c):
                continue

            x = X[c].to_numpy(dtype=float, copy=False)

            try:
                k = kurtosis(x, fisher=False, bias=False)

                if np.isfinite(k) and (k > self.heavy_kurtosis):
                    heavy.append(c)

            except Exception:
                # Conservative fallback:
                # if detection fails, do not classify as heavy-tail
                pass

        return heavy

    # ============================
    # 2) Signed-log transform with percentile clipping
    #    (assumes no NaN/inf)
    # ============================
    def _signed_log_clip(self, x: np.ndarray, col: str, fit: bool):
        x = np.asarray(x, dtype=float)

        if fit:
            lo, hi = np.percentile(
                x,
                [self.clip_low, self.clip_high]
            )

            # Fallback for constant / abnormal columns
            # to avoid lo >= hi
            if (
                (not np.isfinite(lo))
                or (not np.isfinite(hi))
                or (lo >= hi)
            ):
                m = float(np.mean(x))
                eps = 1e-12

                lo, hi = m - eps, m + eps

            self.clip_bounds[col] = (
                float(lo),
                float(hi)
            )

        else:
            lo, hi = self.clip_bounds[col]

        x = np.clip(x, lo, hi)

        return np.sign(x) * np.log1p(np.abs(x))

    # ============================
    # 3) Sigma clipping for ALL columns
    #    (assumes no NaN/inf)
    # ============================
    def _sigma_clip(self, x: np.ndarray, col: str, fit: bool):
        x = np.asarray(x, dtype=float)

        if fit:
            m = float(np.mean(x))
            sd = float(np.std(x)) + 1e-12

            self.sigma_bounds[col] = (m, sd)

        else:
            m, sd = self.sigma_bounds[col]

        lo = m - self.sigma_clip * sd
        hi = m + self.sigma_clip * sd

        return np.clip(x, lo, hi)

    # ============================
    # 4) Fit on training sample
    # ============================
    def fit(self, df: pd.DataFrame, feature_cols):
        """
        Assumes df[feature_cols] has already been cleaned
        externally for NaN / inf values.

        This function performs:
        1) Heavy-tail processing:
           percentile clipping + signed-log transform
           (fit bounds on training data only)

        2) Sigma clipping for all features:
           mean ± sigma_clip * std
           (fit mean/std on training data only)

        3) Scaler fitting on training data only
        """
        assert df.index.is_monotonic_increasing, (
            "df must be sorted in ascending time order"
        )

        assert df.index.is_unique, (
            "df index must be unique"
        )

        assert isinstance(df.index, pd.DatetimeIndex), (
            "df.index must be a DatetimeIndex"
        )

        self.feature_cols = list(feature_cols)

        X = df[self.feature_cols].copy()

        self._check_no_nan_inf(X, where="fit")

        # ===== Heavy-tail columns:
        # auto-detected + manually specified =====
        auto_heavy = self._detect_heavy(X)

        manual_heavy = [
            c for c in self.log_candidates
            if c in self.feature_cols
        ]

        self.heavy_cols = sorted(
            set(auto_heavy + manual_heavy)
        )

        # ===== Apply transforms on TRAIN only
        #       to learn transformation bounds =====
        Xp = X.copy()

        for col in self.heavy_cols:
            Xp[col] = self._signed_log_clip(
                Xp[col].to_numpy(copy=False),
                col,
                fit=True
            )

        for col in self.feature_cols:
            Xp[col] = self._sigma_clip(
                Xp[col].to_numpy(copy=False),
                col,
                fit=True
            )

        self.scaler = (
            RobustScaler()
            if self.use_robust
            else StandardScaler()
        )

        self.scaler.fit(Xp[self.feature_cols])

        self.is_fitted = True

        return self

    # ============================
    # 5) Transform validation / test sample
    # ============================
    def transform(self, df: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError(
                "Call fit(train) before transform(test)."
            )

        assert df.index.is_monotonic_increasing, (
            "df must be sorted in ascending time order"
        )

        assert df.index.is_unique, (
            "df index must be unique"
        )

        assert isinstance(df.index, pd.DatetimeIndex), (
            "df.index must be a DatetimeIndex"
        )

        X = df[self.feature_cols].copy()

        self._check_no_nan_inf(X, where="transform")

        Xp = X.copy()

        for col in self.heavy_cols:
            Xp[col] = self._signed_log_clip(
                Xp[col].to_numpy(copy=False),
                col,
                fit=False
            )

        for col in self.feature_cols:
            Xp[col] = self._sigma_clip(
                Xp[col].to_numpy(copy=False),
                col,
                fit=False
            )

        Z = self.scaler.transform(
            Xp[self.feature_cols]
        )

        out = df.copy()

        out[self.feature_cols] = Z

        return out

    # ============================
    # Print preprocessing statistics
    # ============================
    def report(self, df: pd.DataFrame):
        X = df[self.feature_cols]

        stats = X.describe().T

        print(stats[["mean", "std", "min", "max"]])

        print("\nMax abs:", X.abs().to_numpy().max())
