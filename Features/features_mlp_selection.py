# ============================================
# features/mlp_selection.py
# MLP feature selection
# Logic aligned with tree selection
# ============================================

import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
)


class MLPClassifier(nn.Module):
    """
    MLP for tabular factors.

    Input : [batch_size, n_features]
    Output: [batch_size] logits
    """

    def __init__(
        self,
        input_dim,
        hidden_sizes=(128, 64),
        dropout=0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())

            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def _predict_proba_mlp(
    model,
    X,
    device=None,
):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    X = np.asarray(X, dtype=np.float32)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        logits = model(X_t).detach().cpu().numpy()
        prob = _sigmoid_np(logits)

    proba_2col = np.vstack([1.0 - prob, prob]).T

    model.to("cpu")

    return proba_2col


def _fit_mlp_binary_classifier(
    X_tr,
    y_tr,
    input_dim,
    direction_name="UP",
    n_epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    hidden_sizes=(128, 64),
    dropout=0.1,
    patience=10,
    inner_valid_ratio=0.2,
    random_state=2025,
    device=None,
    verbose=False,
):
    """
    Fit MLP only on the rolling training sample.

    The inner validation split is used only for MLP early stopping.
    It never uses the rolling OOS validation window.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_tr = np.asarray(X_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.float32)

    n = len(X_tr)

    if n <= 2:
        raise ValueError("Not enough samples to train MLP.")

    # ============================================
    # Inner chronological split for early stopping
    # only within training sample
    # ============================================
    if inner_valid_ratio is not None and inner_valid_ratio > 0:

        inner_split = int(n * (1.0 - inner_valid_ratio))
        inner_split = max(1, min(inner_split, n - 1))

        X_fit = X_tr[:inner_split]
        y_fit = y_tr[:inner_split]

        X_es = X_tr[inner_split:]
        y_es = y_tr[inner_split:]

    else:

        X_fit = X_tr
        y_fit = y_tr

        X_es = X_tr
        y_es = y_tr

    train_ds = TensorDataset(
        torch.from_numpy(X_fit),
        torch.from_numpy(y_fit),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    pos_ratio = y_fit.mean() if len(y_fit) else 0.5

    if 0 < pos_ratio < 1:
        pos_weight = torch.tensor(
            (1.0 - pos_ratio) / (pos_ratio + 1e-8),
            dtype=torch.float32,
            device=device,
        )
    else:
        pos_weight = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    def eval_inner_metric():
        model.eval()

        with torch.no_grad():
            X_es_t = torch.from_numpy(X_es).to(device)
            logits = model(X_es_t).detach().cpu().numpy()

        prob = _sigmoid_np(logits)

        if len(np.unique(y_es)) > 1:
            try:
                return roc_auc_score(y_es, prob)
            except Exception:
                pass

        try:
            proba_2col = np.vstack([1.0 - prob, prob]).T
            return -log_loss(
                y_es,
                proba_2col,
                labels=[0, 1],
            )
        except Exception:
            return -np.inf

    best_metric = -np.inf
    best_state = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

        metric = eval_inner_metric()

        if metric > best_metric + 1e-6:
            best_metric = metric
            best_state = copy.deepcopy(
                model.state_dict()
            )
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"[{direction_name}] "
                f"epoch={epoch}, inner_metric={metric:.4f}"
            )

        if patience is not None and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to("cpu")

    return model


def train_one_direction_mlp(
    df,
    feature_cols,
    target_col,
    direction_name="UP",

    n_epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    hidden_sizes=(128, 64),
    dropout=0.1,
    patience=10,
    inner_valid_ratio=0.2,
    device=None,

    valid_size=250,
    n_windows=3,

    imp_min=0.0,
    imp_quantile=0.6,
    top_k_max=25,
    auc_single_min=0.55,
    n_perm=5,
    max_single_check=25,

    verbose=False,
):
    """
    Train one-direction MLP model and select features.

    This function is aligned with:
    - train_one_direction_xgb
    - train_one_direction_rf
    - train_one_direction_lgbm
    - train_one_direction_catboost

    The only intended difference is the core algorithm:
    MLPClassifier instead of tree models.
    """

    print(
        f"\n========== Start training {direction_name} "
        f"(MLP), target = {target_col} ==========\n"
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

    col_to_j = {
        c: i for i, c in enumerate(feature_cols)
    }

    def has_both_classes(y_arr) -> bool:
        return len(np.unique(y_arr)) > 1

    # ============================================
    # 2. Decide permutation-importance metric
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

    imp_by_window = []
    window_metrics = []

    input_dim = len(feature_cols)

    # ============================================
    # 3. Rolling OOS training / validation
    # ============================================
    for w, (tr0, tr1, va0, va1) in enumerate(windows, 1):

        train_df = df.iloc[tr0:tr1]
        valid_df = df.iloc[va0:va1]

        X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_tr = train_df[target_col].to_numpy().astype(int)

        X_va = valid_df[feature_cols].to_numpy(dtype=np.float32)
        y_va = valid_df[target_col].to_numpy().astype(int)

        pos_ratio_tr = y_tr.mean() if len(y_tr) else np.nan

        model_w = _fit_mlp_binary_classifier(
            X_tr=X_tr,
            y_tr=y_tr,
            input_dim=input_dim,
            direction_name=f"{direction_name}-W{w}",
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            patience=patience,
            inner_valid_ratio=inner_valid_ratio,
            random_state=2025 + 17 * w,
            device=device,
            verbose=verbose,
        )

        proba_va = _predict_proba_mlp(
            model_w,
            X_va,
            device=device,
        )

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

                proba_p = _predict_proba_mlp(
                    model_w,
                    Xp,
                    device=device,
                )

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
    # 4. Aggregate OOS permutation importance
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
    # 5. Train final model using latest rolling split
    # ============================================
    tr0, tr1, va0, va1 = windows[-1]

    train_df = df.iloc[tr0:tr1]
    valid_df = df.iloc[va0:va1]

    X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_tr = train_df[target_col].to_numpy().astype(int)

    X_va = valid_df[feature_cols].to_numpy(dtype=np.float32)
    y_va = valid_df[target_col].to_numpy().astype(int)

    model = _fit_mlp_binary_classifier(
        X_tr=X_tr,
        y_tr=y_tr,
        input_dim=input_dim,
        direction_name=f"{direction_name}-FINAL",
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        patience=patience,
        inner_valid_ratio=inner_valid_ratio,
        random_state=2025,
        device=device,
        verbose=verbose,
    )

    proba_va = _predict_proba_mlp(
        model,
        X_va,
        device=device,
    )

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
    # 6. Importance threshold
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
    # 7. Single-factor OOS AUC check
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

                try:
                    single_model = _fit_mlp_binary_classifier(
                        X_tr=X_tr_w,
                        y_tr=y_tr_w,
                        input_dim=1,
                        direction_name=(
                            f"{direction_name}-single-{col}-W{w}"
                        ),
                        n_epochs=max(50, n_epochs // 2),
                        batch_size=batch_size,
                        lr=lr,
                        weight_decay=weight_decay,
                        hidden_sizes=(32, 16),
                        dropout=dropout,
                        patience=patience,
                        inner_valid_ratio=inner_valid_ratio,
                        random_state=2025 + 31 * w + 7 * j,
                        device=device,
                        verbose=False,
                    )

                    proba = _predict_proba_mlp(
                        single_model,
                        X_va_w,
                        device=device,
                    )

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
    # 8. Final feature selection
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
