"""Per-recipe quantities (guide Ch. 9), each with its seed standard deviation (ddof = 1).

Seeds that went non-finite in the final stage are excluded from every statistic of their recipe and counted in
`n_nonfinite_seeds` (the guide does not cover this case; it is reported, never silently dropped).
"""
import numpy as np
import pandas as pd

from capacity_study.analysis.load import RecipeRaw
from capacity_study.config import StudyConfig


def t1(preds: np.ndarray) -> float:
    """Mean over windows of the across-seed prediction variance. preds: (S, n)."""
    return float(preds.var(axis=0, ddof=1).mean())


def t1_logratio(preds_train: np.ndarray, preds_test: np.ndarray) -> float:
    return 0.5 * float(np.log(t1(preds_test) / t1(preds_train)))


def jackknife_se(preds_train: np.ndarray, preds_test: np.ndarray) -> float:
    s = preds_train.shape[0]
    loo = np.array([t1_logratio(np.delete(preds_train, i, 0), np.delete(preds_test, i, 0)) for i in range(s)])
    return float(np.sqrt((s - 1) / s * ((loo - loo.mean()) ** 2).sum()))


def _sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def recipe_tables(cfg: StudyConfig, raws: list[RecipeRaw]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-recipe table, per-seed table). Untunable recipes get the configured sentinels."""
    blk = cfg.analysis.v1_block
    a = cfg.analysis
    rows, seed_rows = [], []
    for raw in raws:
        r = raw.recipe
        base = {"recipe_id": r.recipe_id, "depth": r.depth, "width": r.width, "regularization": r.regularization,
                "tier": r.tier, "history": r.history, "final_job": raw.final_job, "n_train": raw.n_train,
                "batch_size": raw.batch_size, "untunable": raw.untunable,
                "tuned_lr": raw.tuning.best_lr, "tuned_wd": raw.tuning.best_wd,
                "tuned_lr_x_wd": (raw.tuning.best_lr * raw.tuning.best_wd) if not raw.untunable else None,
                "edge_hit_lr": raw.tuning.edge_hit.get("lr", False), "edge_hit_wd": raw.tuning.edge_hit.get("wd", False),
                "infeasible_fraction": raw.tuning.infeasible_fraction}
        if raw.untunable:
            rows.append(base | {"n_seeds": 0, "n_nonfinite_seeds": 0,
                                "test_r2_mean": a.sentinel_test_r2, "test_r2_sd": 0.0,
                                "train_r2_mean": a.sentinel_test_r2, "train_r2_sd": 0.0,
                                "gap_mean": a.sentinel_gap, "gap_sd": 0.0,
                                "v1a_abs_mean": a.sentinel_v1, "v1a_sd": 0.0,
                                "v1b_abs_mean": a.sentinel_v1, "v1b_sd": 0.0,
                                "t1_train": float("nan"), "t1_test": float("nan"),
                                "t1_logratio": a.sentinel_t1_logratio, "t1_logratio_se": 0.0,
                                "spread_mean": float("nan"), "spread_sd": float("nan"),
                                "rel_weight_change_mean": float("nan"), "rel_weight_change_sd": float("nan")})
            continue
        ok = raw.finite
        tr, te = raw.train_curve[ok], raw.test_curve[ok]
        p_tr, p_te = raw.preds_train[ok], raw.preds_test[ok]
        train_r2 = tr[:, -blk:].mean(axis=1)
        test_r2 = te[:, -blk:].mean(axis=1)
        v1a = train_r2 - tr[:, -2 * blk:-blk].mean(axis=1)
        v1b = test_r2 - te[:, -2 * blk:-blk].mean(axis=1)
        gap = train_r2 - test_r2
        spread = p_te.std(axis=1, ddof=1)
        rwc = raw.rel_weight_change[ok]
        enough = ok.sum() >= 2
        rows.append(base | {
            "n_seeds": int(ok.sum()), "n_nonfinite_seeds": int((~ok).sum()),
            "test_r2_mean": float(test_r2.mean()), "test_r2_sd": _sd(test_r2),
            "train_r2_mean": float(train_r2.mean()), "train_r2_sd": _sd(train_r2),
            "gap_mean": float(gap.mean()), "gap_sd": _sd(gap),
            "v1a_abs_mean": float(abs(v1a.mean())), "v1a_sd": _sd(v1a),
            "v1b_abs_mean": float(abs(v1b.mean())), "v1b_sd": _sd(v1b),
            "t1_train": t1(p_tr) if enough else float("nan"), "t1_test": t1(p_te) if enough else float("nan"),
            "t1_logratio": t1_logratio(p_tr, p_te) if enough else float("nan"),
            "t1_logratio_se": jackknife_se(p_tr, p_te) if ok.sum() >= 3 else float("nan"),
            "spread_mean": float(spread.mean()), "spread_sd": _sd(spread),
            "rel_weight_change_mean": float(rwc.mean()), "rel_weight_change_sd": _sd(rwc)})
        seeds = np.flatnonzero(ok)
        for j, s in enumerate(seeds):
            seed_rows.append({"recipe_id": r.recipe_id, "seed": int(s), "train_r2": float(train_r2[j]),
                              "test_r2": float(test_r2[j]), "gap": float(gap[j]), "v1a": float(v1a[j]),
                              "v1b": float(v1b[j]), "spread": float(spread[j]), "rel_weight_change": float(rwc[j])})
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def check_sentinels(cfg: StudyConfig, table: pd.DataFrame) -> None:
    """Every real value must be at least 10x smaller in magnitude than its sentinel."""
    a = cfg.analysis
    real = table[~table["untunable"]]
    for col, sentinel in (("test_r2_mean", a.sentinel_test_r2), ("gap_mean", a.sentinel_gap),
                          ("t1_logratio", a.sentinel_t1_logratio), ("v1a_abs_mean", a.sentinel_v1),
                          ("v1b_abs_mean", a.sentinel_v1)):
        worst = real[col].abs().max()
        if np.isfinite(worst) and worst * 10 > abs(sentinel):
            raise ValueError(f"check_sentinels: real {col} reaches {worst}, too close to sentinel {sentinel}")
