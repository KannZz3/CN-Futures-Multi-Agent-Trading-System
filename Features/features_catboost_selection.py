# ============================================
# features/catboost_selection.py
# CatBoost feature selection
# Logic aligned with xgb_selection.py / rf_selection.py / lgbm_selection.py
# ============================================

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
)


def train_one_direction_catboost(
    df,
    feature_cols,
    target_col,
    direction_name="UP",

    n_estimators=400,
    max_depth=5,
    min_samples_leaf=50,
    max_features="sqrt",

    valid_size=250,
    n_windows=3,

    imp_min=0.0,
    imp_quantile=0.6,
    top_k_max=25,
    auc_single_min=0.55,
    n_perm=5,
    max_single_check=25,
):
    """
    Train one-direction CatBoost model and select features.

    This function is designed to be parallel to:
    - train_one_direction_xgb
    - train_one_direction_rf
    - train_one_direction_lgbm

    The only intended difference is the core tree algorithm:
    CatBoostClassifier.
    """

    print(
        f"\n========== Start training {direction_name} "
        f"(CatBoost), target = {target_col} ==========\n"
    )

    df = df.sort_index()
    n = len(df)

    if n <= valid_size:
        raise ValueError(
            f"Need n > valid_size. "
            f"Got n={n}, valid_size={valid_size}."
        )

    max_w = (n - 1) // valid_size

    if n_windows > max_w:
        print(
            f"[Warn] n_windows={n_windows} too large "
            f"for n={n}, valid_size={valid_size}. "
            f"Capping to {max_w}."
        )
        n_windows = max_w

    # ============================================
    # 1. Construct rolling train/validation windows
    # ============================================
    windows = []

    for w in range(n_windows, 0, -1):

        train_end = n - w * valid_size

        valid_start = train_end
        valid_end = train_end + valid_size

        if train_end <= 0:
            continue

        windows.append(
            (0, train_end, valid_start, valid_end)
        )

    if len(windows) == 0:
        raise RuntimeError(
            "Failed to construct windows. "
            "Check n/valid_size/n_windows."
        )

    print(f"[Info] Using {len(windows)} window(s):")

    for k, (tr0, tr1, va0, va1) in enumerate(windows, 1):
        print(
            f"  W{k}: "
            f"train[{tr0}:{tr1}] ({tr1 - tr0}), "
            f"valid[{va0}:{va1}] ({va1 - va0})"
        )

    # ============================================
    # 2. Convert max_features into CatBoost rsm
    # ============================================
    d = len(feature_cols)

    if isinstance(max_features, str):

        if max_features == "sqrt":
            rsm = (
                float(np.sqrt(d) / d)
                if d > 0 else 1.0
            )

        elif max_features == "log2":
            rsm = (
                float(np.log2(d) / d)
                if d > 0 else 1.0
            )

        else:
            rsm = 1.0

    else:
        rsm = float(max_features)

    rsm = float(
        np.clip(rsm, 1e-6, 1.0)
    )

    col_to_j = {
        c: i for i, c in enumerate(feature_cols)
    }

    def has_both_classes(y_arr) -> bool:
        return len(np.unique(y_arr)) > 1

    # ============================================
    # 3. Decide permutation-importance metric
    # ============================================
    use_auc = True

    for (_, _, va0, va1) in windows:
        y_va = (
            df.iloc[va0:va1][target_col]
            .to_numpy()
            .astype(int)
        )

        if not has_both_classes(y_va):
            use_auc = False
            break

    metric_mode = "auc" if use_auc else "neg_logloss"

    print(
        f"\n[Info] metric_mode for "
        f"permutation importance = {metric_mode}"
    )

    def metric_score(y_true, proba_2col):

        if metric_mode == "auc":
            return roc_auc_score(
                y_true,
                proba_2col[:, 1]
            )

        return -log_loss(
            y_true,
            proba_2col,
            labels=[0, 1],
        )

    # ============================================
    # 4. Build CatBoost model
    # ============================================
    def build_model(
        scale_pos_weight: float,
        random_state: int,
    ):
        return CatBoostClassifier(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=0.05,

            l2_leaf_reg=1.0,
            min_data_in_leaf=int(min_samples_leaf),

            rsm=rsm,

            loss_function="Logloss",
            eval_metric="Logloss",

            random_seed=random_state,
            thread_count=-1,

            scale_pos_weight=scale_pos_weight,

            verbose=False,
            allow_writing_files=False,
        )

    imp_by_window = []
    window_metrics = []

    # ============================================
    # 5. Rolling OOS training / validation
    # ============================================
    for w, (tr0, tr1, va0, va1) in enumerate(windows, 1):

        train_df = df.iloc[tr0:tr1]
        valid_df = df.iloc[va0:va1]

        X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_tr = train_df[target_col].to_numpy().astype(int)

        X_va = valid_df[feature_cols].to_numpy(dtype=np.float32)
        y_va = valid_df[target_col].to_numpy().astype(int)

        pos_ratio_tr = y_tr.mean() if len(y_tr) else np.nan

        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()

        scale_pos_weight = (
            n_neg / max(n_pos, 1)
            if n_pos > 0 else 1.0
        )

        model_w = build_model(
            scale_pos_weight=scale_pos_weight,
            random_state=2025 + 17 * w,
        )

        model_w.fit(X_tr, y_tr)

        proba_va = model_w.predict_proba(X_va)
        baseline = metric_score(y_va, proba_va)

        prob_va = proba_va[:, 1]
        pred_va = (prob_va > 0.5).astype(int)

        acc_va = accuracy_score(y_va, pred_va)

        auc_va = (
            roc_auc_score(y_va, prob_va)
            if has_both_classes(y_va)
            else np.nan
        )

        window_metrics.append({
            "window": w,
            "train_len": len(train_df),
            "valid_len": len(valid_df),
            "pos_ratio_train": pos_ratio_tr,
            "acc_valid": acc_va,
            "auc_valid": auc_va,
            "baseline_metric": baseline,
            "metric_mode": metric_mode,
        })

        # ============================================
        # OOS permutation importance
        # ============================================
        rng = np.random.RandomState(
            2025 + 1000 * w
        )

        perm_importance = {}

        for j, col in enumerate(feature_cols):

            deltas = []

            for _ in range(n_perm):

                Xp = X_va.copy()

                shuffled = Xp[:, j].copy()
                rng.shuffle(shuffled)
                Xp[:, j] = shuffled

                proba_p = model_w.predict_proba(Xp)

                metric_p = metric_score(
                    y_va,
                    proba_p,
                )

                deltas.append(
                    baseline - metric_p
                )

            perm_importance[col] = float(
                np.mean(deltas)
            )

        imp_ser = (
            pd.Series(perm_importance)
            .sort_values(ascending=False)
        )

        imp_by_window.append(imp_ser)

        print(f"\n--- W{w} OOS valid metrics ---")
        print(
            f"train_len={len(train_df)}, "
            f"valid_len={len(valid_df)}, "
            f"pos_ratio_train={pos_ratio_tr:.4f}, "
            f"valid_ACC={acc_va:.4f}, "
            f"valid_AUC={auc_va:.4f}"
        )

        print(
            f"W{w} OOS permutation importance "
            f"(Top 10):"
        )
        print(imp_ser.head(10))

    # ============================================
    # 6. Aggregate OOS permutation importance
    # ============================================
    imp_mat = pd.concat(
        imp_by_window,
        axis=1,
    )

    imp_mean = imp_mat.mean(axis=1)
    imp_median = imp_mat.median(axis=1)
    imp_pos_frac = (imp_mat > 0).mean(axis=1)

    importance_series = (
        imp_mean.sort_values(ascending=False)
    )

    print(
        f"\n[{direction_name}] "
        f"Aggregated OOS permutation importance "
        f"(mean) Top 20:"
    )

    top20 = importance_series.head(20).index

    print(pd.DataFrame({
        "imp_mean": imp_mean.loc[top20],
        "imp_median": imp_median.loc[top20],
        "pos_frac": imp_pos_frac.loc[top20],
    }))

    # ============================================
    # 7. Train final model using latest rolling split
    # ============================================
    tr0, tr1, va0, va1 = windows[-1]

    train_df = df.iloc[tr0:tr1]
    valid_df = df.iloc[va0:va1]

    X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_tr = train_df[target_col].to_numpy().astype(int)

    X_va = valid_df[feature_cols].to_numpy(dtype=np.float32)
    y_va = valid_df[target_col].to_numpy().astype(int)

    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()

    scale_pos_weight = (
        n_neg / max(n_pos, 1)
        if n_pos > 0 else 1.0
    )

    model = build_model(
        scale_pos_weight=scale_pos_weight,
        random_state=2025,
    )

    model.fit(X_tr, y_tr)

    proba_va = model.predict_proba(X_va)
    prob_va = proba_va[:, 1]
    pred_va = (prob_va > 0.5).astype(int)

    acc = accuracy_score(y_va, pred_va)

    auc = (
        roc_auc_score(y_va, prob_va)
        if has_both_classes(y_va)
        else np.nan
    )

    print(
        f"\n{direction_name} - FINAL OOS "
        f"(train={len(train_df)}/valid={len(valid_df)}) "
        f"ACC: {acc:.4f}"
    )

    print(
        f"{direction_name} - FINAL OOS "
        f"AUC:  {auc:.4f}"
    )

    print(
        f"\n{direction_name} - "
        f"FINAL OOS classification_report:"
    )

    print(classification_report(y_va, pred_va))

    # ============================================
    # 8. Built-in CatBoost importance
    # ============================================
    gini_series = (
        pd.Series(
            model.get_feature_importance(),
            index=feature_cols,
        )
        .sort_values(ascending=False)
    )

    # ============================================
    # 9. Importance threshold
    # ============================================
    thr_abs = -np.inf if imp_min is None else imp_min

    thr_quantile = (
        -np.inf
        if imp_quantile is None
        else imp_mean.quantile(imp_quantile)
    )

    thr = max(thr_abs, thr_quantile)

    print(
        f"\n{direction_name} - "
        f"OOS perm-imp threshold: "
        f"max(imp_min={thr_abs:.6f}, "
        f"quantile={thr_quantile:.6f}) "
        f"= {thr:.6f}"
    )

    imp_candidates = (
        imp_mean[imp_mean >= thr]
        .sort_values(ascending=False)
        .index.tolist()
    )

    print(
        f"{direction_name} - "
        f"features passing OOS importance "
        f"threshold: {len(imp_candidates)}"
    )

    # ============================================
    # 10. Single-factor OOS AUC check
    # ============================================
    single_check_features = (
        imp_candidates[:max_single_check]
        if max_single_check is not None
        else imp_candidates
    )

    print(
        f"{direction_name} - Running "
        f"single-factor OOS AUC check "
        f"for top {len(single_check_features)} "
        f"features"
    )

    single_auc_by_window = {
        col: [] for col in single_check_features
    }

    if (
        auc_single_min is not None
        and len(single_check_features) > 0
    ):

        print(
            f"\n{direction_name} - "
            f"Single-factor OOS AUC check "
            f"(threshold = {auc_single_min:.3f})"
        )

        for col in single_check_features:

            j = col_to_j[col]

            for w, (tr0, tr1, va0, va1) in enumerate(windows, 1):

                train_df_w = df.iloc[tr0:tr1]
                valid_df_w = df.iloc[va0:va1]

                X_tr_w = (
                    train_df_w[feature_cols]
                    .to_numpy(dtype=np.float32)[:, [j]]
                )

                y_tr_w = (
                    train_df_w[target_col]
                    .to_numpy()
                    .astype(int)
                )

                X_va_w = (
                    valid_df_w[feature_cols]
                    .to_numpy(dtype=np.float32)[:, [j]]
                )

                y_va_w = (
                    valid_df_w[target_col]
                    .to_numpy()
                    .astype(int)
                )

                n_pos_w = (y_tr_w == 1).sum()
                n_neg_w = (y_tr_w == 0).sum()

                spw_w = (
                    n_neg_w / max(n_pos_w, 1)
                    if n_pos_w > 0 else 1.0
                )

                single_cat = CatBoostClassifier(
                    iterations=min(200, n_estimators),
                    depth=max_depth,
                    learning_rate=0.05,

                    l2_leaf_reg=1.0,
                    min_data_in_leaf=int(min_samples_leaf),

                    rsm=1.0,

                    loss_function="Logloss",
                    eval_metric="Logloss",

                    random_seed=2025 + 31 * w + 7 * j,
                    thread_count=-1,

                    scale_pos_weight=spw_w,

                    verbose=False,
                    allow_writing_files=False,
                )

                try:
                    single_cat.fit(X_tr_w, y_tr_w)

                    proba = single_cat.predict_proba(X_va_w)

                    auc_w = (
                        roc_auc_score(
                            y_va_w,
                            proba[:, 1],
                        )
                        if has_both_classes(y_va_w)
                        else np.nan
                    )

                except Exception:
                    auc_w = np.nan

                single_auc_by_window[col].append(auc_w)

        single_auc_mean = (
            pd.Series({
                col: (
                    float(np.nanmean(aucs))
                    if len(aucs)
                    else np.nan
                )
                for col, aucs in single_auc_by_window.items()
            })
            .sort_values(ascending=False)
        )

        single_auc_min = pd.Series({
            col: (
                float(np.nanmin(aucs))
                if np.any(~np.isnan(aucs))
                else np.nan
            )
            for col, aucs in single_auc_by_window.items()
        })

        print(
            f"\n{direction_name} - "
            f"Single-factor OOS AUC mean "
            f"(Top 20):"
        )

        top20 = single_auc_mean.head(20).index

        print(pd.DataFrame({
            "auc_mean": single_auc_mean.loc[top20],
            "auc_min": single_auc_min.loc[top20],
        }))

    else:

        single_auc_mean = pd.Series({
            col: np.nan
            for col in feature_cols
        })

        single_auc_min = pd.Series({
            col: np.nan
            for col in feature_cols
        })

    # ============================================
    # 11. Final feature selection
    # ============================================
    base_candidates = single_check_features

    if auc_single_min is not None:

        selected_features = [
            col
            for col in base_candidates
            if (
                col in single_auc_mean.index
                and not np.isnan(single_auc_mean[col])
                and single_auc_mean[col] >= auc_single_min
            )
        ]

    else:

        selected_features = list(base_candidates)

    if (
        top_k_max is not None
        and len(selected_features) > top_k_max
    ):
        selected_features = selected_features[:top_k_max]

    print(
        f"\n{direction_name} - "
        f"Final selected features: "
        f"{len(selected_features)}"
    )

    print(
        f"{direction_name} - "
        f"Selected list "
        f"(top 30 shown):"
    )

    print(selected_features[:30])

    # ============================================
    # 12. Return result
    # No full-sample prob / pred returned.
    # They are not strictly OOS.
    # ============================================
    y_all = (
        df[target_col]
        .to_numpy()
        .astype(int)
    )

    result = {
        "model": model,

        "top_features": selected_features,
        "selected_features": selected_features,

        "importance_series": importance_series,

        "importance_detail": (
            pd.DataFrame({
                "imp_mean": imp_mean,
                "imp_median": imp_median,
                "pos_frac": imp_pos_frac,
            })
            .sort_values(
                "imp_mean",
                ascending=False,
            )
        ),

        "gini_importance": gini_series,

        "single_auc": single_auc_mean,
        "single_auc_min": single_auc_min,

        "acc": acc,
        "auc": auc,

        "true": y_all,
        "index": df.index,

        "windows": windows,
        "window_metrics": window_metrics,
    }

    return result
