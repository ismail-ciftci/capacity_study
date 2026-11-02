"""Equivalence margins (guide Ch. 8) and recipe status (Ch. 10)."""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

VALID, UNSATURATED, OVERFIT, UNTUNABLE = "valid", "failure-unsaturated", "failure-overfit", "untunable"
STATUS_LABELS = (VALID, UNSATURATED, OVERFIT, UNTUNABLE)


@dataclass(frozen=True)
class Margins:
    n: int
    alpha: float
    beta: float
    delta_mu: float      # in units of seed sigma
    delta_sigma: float   # on the log standard-deviation ratio


def margins(n: int, alpha: float, beta: float) -> Margins:
    df = 2 * (n - 1)
    d_mu = (stats.t.ppf(1 - alpha, df) + stats.t.ppf(1 - beta / 2, df)) * math.sqrt(2 / n)
    d_sigma = (stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta / 2)) / math.sqrt(n - 1)
    return Margins(n, alpha, beta, float(d_mu), float(d_sigma))


def flatness_threshold(table: pd.DataFrame, m: Margins) -> float:
    """delta_mu x median over (tunable) recipes of the seed sd of test R2: one value for V1a and V1b."""
    sds = table.loc[~table["untunable"], "test_r2_sd"].dropna()
    if sds.empty:
        raise ValueError("flatness_threshold: no tunable recipe with a seed sd of test R2")
    return float(m.delta_mu * np.median(sds))


def classify(table: pd.DataFrame, flat_thr: float, m: Margins) -> pd.Series:
    """Precedence: untunable; unsaturated (either slope above threshold); overfit (T1 log ratio > delta_sigma)."""
    def one(row) -> str:
        if row["untunable"]:
            return UNTUNABLE
        if row["v1a_abs_mean"] > flat_thr or row["v1b_abs_mean"] > flat_thr:
            return UNSATURATED
        if row["t1_logratio"] > m.delta_sigma:
            return OVERFIT
        return VALID
    return table.apply(one, axis=1)
