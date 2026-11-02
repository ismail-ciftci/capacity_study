"""Population characteristics (guide Ch. 11): failure rates, data saturation, functional ANOVA, lr-vs-batch."""
import itertools

import numpy as np
import pandas as pd

from capacity_study.analysis.status import STATUS_LABELS
from capacity_study.config import StudyConfig

FACTORS = ("depth", "width", "regularization", "tier", "history")


def failure_rates(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Status composition per option and per pair of options (counts and fractions)."""
    def composition(group: pd.DataFrame) -> dict:
        counts = group["status"].value_counts()
        n = len(group)
        return {"n": n} | {f"n_{s}": int(counts.get(s, 0)) for s in STATUS_LABELS} \
            | {f"frac_{s}": counts.get(s, 0) / n for s in STATUS_LABELS}

    single = [{"factor": f, "option": opt} | composition(g) for f in FACTORS for opt, g in table.groupby(f)]
    pairs = [{"factor_a": fa, "option_a": oa, "factor_b": fb, "option_b": ob} | composition(g)
             for fa, fb in itertools.combinations(FACTORS, 2) for (oa, ob), g in table.groupby([fa, fb])]
    return pd.DataFrame(single), pd.DataFrame(pairs)


def data_saturation(cfg: StudyConfig, table: pd.DataFrame, delta_mu: float) -> pd.DataFrame:
    """Consecutive data-size steps along history (within each tier) and along the nested financial tiers 1-4
    (within each history). Tier 5 is not on the size axis. Differences are point estimates; `within_resolution`
    compares |difference| with delta_mu x pooled seed sd (the map resolution), without a test."""
    rows = []
    axes = {"history": (list(cfg.grid.histories), ["depth", "width", "regularization", "tier"]),
            "tier": ([t for t in cfg.grid.tiers if t in (1, 2, 3, 4)], ["depth", "width", "regularization", "history"])}
    indexed = table.set_index(list(FACTORS))
    for axis, (levels, fixed) in axes.items():
        for combo, _ in table.groupby(fixed):
            key = dict(zip(fixed, combo))
            for lo, hi in zip(levels, levels[1:]):
                a = indexed.loc[tuple(key.get(f, lo) if f != axis else lo for f in FACTORS)]
                b = indexed.loc[tuple(key.get(f, hi) if f != axis else hi for f in FACTORS)]
                row = {"axis": axis, **key, "from": lo, "to": hi, "status_from": a["status"],
                       "status_to": b["status"], "untunable": bool(a["untunable"] or b["untunable"])}
                for q in ("test_r2", "gap"):
                    d = b[f"{q}_mean"] - a[f"{q}_mean"]
                    sp = np.sqrt((a[f"{q}_sd"] ** 2 + b[f"{q}_sd"] ** 2) / 2)
                    row |= {f"d_{q}": d, f"d_{q}_std": d / sp if sp > 0 else np.nan,
                            f"{q}_within_resolution": bool(abs(d) < delta_mu * sp)}
                row["d_t1_logratio"] = b["t1_logratio"] - a["t1_logratio"]
                rows.append(row)
    return pd.DataFrame(rows)


def fanova(cfg: StudyConfig, table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Functional ANOVA over the full grid from cell values (no surrogate): variance share of every main and
    interaction effect. Requires a value for every recipe."""
    levels = [list(getattr(cfg.grid, name)) for name in ("depths", "widths", "regularizations", "tiers", "histories")]
    shape = tuple(len(lv) for lv in levels)
    f = np.full(shape, np.nan)
    for row in table.itertuples():
        f[tuple(lv.index(getattr(row, fac)) for lv, fac in zip(levels, FACTORS))] = getattr(row, column)
    if np.isnan(f).any():
        raise ValueError(f"fanova: {int(np.isnan(f).sum())} grid cells lack {column}")
    total = f.var()
    effects: dict[tuple[int, ...], np.ndarray] = {(): np.full(shape, f.mean())}
    rows = []
    for order in range(1, len(FACTORS) + 1):
        for subset in itertools.combinations(range(len(FACTORS)), order):
            others = tuple(i for i in range(len(FACTORS)) if i not in subset)
            marginal = f.mean(axis=others, keepdims=True) if others else f
            effect = np.broadcast_to(marginal, shape).copy()
            for k in range(order):
                for sub in itertools.combinations(subset, k):
                    effect -= effects[sub]
            effects[subset] = effect
            v = float((effect ** 2).mean())
            rows.append({"effect": " x ".join(FACTORS[i] for i in subset), "order": order, "variance": v,
                         "share": v / total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def lr_vs_batch(table: pd.DataFrame) -> pd.DataFrame:
    cols = ["recipe_id", *FACTORS, "batch_size", "tuned_lr", "tuned_wd", "tuned_lr_x_wd", "edge_hit_lr",
            "edge_hit_wd", "infeasible_fraction", "untunable"]
    return table[cols].copy()
