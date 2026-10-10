from __future__ import annotations

import numpy as np
import pandas as pd


def ewma(score: pd.Series, alpha: float) -> pd.Series:
    if not 0 < alpha <= 1:
        raise ValueError("EWMA alpha must be between 0 and 1.")
    return score.ewm(alpha=alpha, adjust=False).mean().rename("ewma")


def cusum(score: pd.Series, center: float, scale: float, reference_shift: float) -> pd.Series:
    if not np.isfinite([center, scale, reference_shift]).all() or scale <= 0 or reference_shift <= 0:
        raise ValueError("CUSUM center/scale must be finite and scale/reference shift positive.")
    state = 0.0
    values = []
    for value in score:
        if not np.isfinite(value):
            raise ValueError("CUSUM observations must be finite.")
        state = max(0.0, state + (value - center) / scale - reference_shift)
        values.append(state)
    return pd.Series(values, index=score.index, name="cusum")
