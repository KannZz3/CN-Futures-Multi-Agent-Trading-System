# -*- coding: utf-8 -*-
"""Auto-split from the original single-contract notebooks.
Core formulas, thresholds, and function bodies are kept unchanged.
"""

import numpy as np
import pandas as pd

def construct_labels(data: pd.DataFrame, H: int = 5, TH: float = 0.05):
    """
    Construct the same labels used in the original notebooks.
    """
    data = data.copy()

    # 1. Close-to-close return over horizon H
    data["future_ret_H"] = data["close"].shift(-H) / data["close"] - 1

    # Binary labels for large upward / downward moves
    data["y_up"] = (data["future_ret_H"] >= TH).astype(int)
    data["y_down"] = (data["future_ret_H"] <= -TH).astype(int)

    # 4. Drop samples whose future returns are not observable
    mask = data["future_ret_H"].notna()
    data_train = data[mask].copy()

    return data, data_train
