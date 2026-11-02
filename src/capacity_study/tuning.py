"""Tuning protocol: Halton trial points, objective, best-trial selection, edge hits, range freezing.

Trial points come from scipy.stats.qmc.Halton (the engine Optuna's QMCSampler wraps) because QMCSampler samples
the first trial with its random independent sampler when trials are asked before any completes. Results are
recorded into an Optuna study per recipe with add_trial.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from capacity_study.config import StudyConfig


@dataclass(frozen=True)
class Ranges:
    lr: tuple[float, float]
    wd: tuple[float, float]

    def to_dict(self) -> dict:
        return {"lr": list(self.lr), "wd": list(self.wd)}

    @staticmethod
    def from_dict(d: dict) -> "Ranges":
        return Ranges(tuple(d["lr"]), tuple(d["wd"]))


def calibration_ranges(cfg: StudyConfig) -> Ranges:
    return Ranges(cfg.tuning.calibration_lr, cfg.tuning.calibration_wd)


def trial_points(cfg: StudyConfig, regularization: str, ranges: Ranges) -> list[dict[str, float]]:
    """n_trials points; 1-D (lr) without weight decay, 2-D (lr, wd) with it; log-uniform mapping."""
    dims = [("lr", ranges.lr)] + ([("wd", ranges.wd)] if regularization == "wd" else [])
    unit = qmc.Halton(d=len(dims), scramble=True, seed=cfg.tuning.halton_seed).random(cfg.tuning.n_trials)
    points = []
    for row in unit:
        p = {"lr": 0.0, "wd": 0.0}
        for u, (name, (lo, hi)) in zip(row, dims):
            p[name] = float(10 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo))))
        points.append(p)
    return points


@dataclass(frozen=True)
class TuningResult:
    recipe_id: int
    regularization: str
    objectives: list[float | None]   # None = infeasible
    best_trial: int | None           # None = untunable (every trial infeasible)
    best_lr: float | None
    best_wd: float | None
    edge_hit: dict[str, bool]
    infeasible_fraction: float

    @property
    def untunable(self) -> bool:
        return self.best_trial is None

    def to_dict(self) -> dict:
        return dict(vars(self)) | {"untunable": self.untunable}


def objective(r2_eval_tail: np.ndarray, finite_all: np.ndarray) -> list[float | None]:
    """r2_eval_tail: (tail_epochs, n_trials); finite_all: (n_trials,) bool over the whole run."""
    means = r2_eval_tail.mean(axis=0)
    return [float(m) if ok and np.isfinite(m) else None for m, ok in zip(means, finite_all)]


def select_best(recipe_id: int, regularization: str, points: list[dict[str, float]],
                objectives: list[float | None], ranges: Ranges) -> TuningResult:
    scores = np.array([-np.inf if o is None else o for o in objectives])
    infeasible = sum(o is None for o in objectives) / len(objectives)
    if np.all(np.isneginf(scores)):
        return TuningResult(recipe_id, regularization, objectives, None, None, None, {}, infeasible)
    best = int(np.argmax(scores))  # first maximum = lowest trial index on ties
    edge = {"lr": _edge_hit(points[best]["lr"], ranges.lr, len(points))}
    if regularization == "wd":
        edge["wd"] = _edge_hit(points[best]["wd"], ranges.wd, len(points))
    return TuningResult(recipe_id, regularization, objectives, best, points[best]["lr"], points[best]["wd"],
                        edge, infeasible)


def _edge_hit(value: float, bounds: tuple[float, float], n_trials: int) -> bool:
    lo, hi = math.log10(bounds[0]), math.log10(bounds[1])
    spacing = (hi - lo) / n_trials
    v = math.log10(value)
    return v - lo <= spacing or hi - v <= spacing


def freeze_ranges(cfg: StudyConfig, results: list[TuningResult]) -> tuple[Ranges, dict]:
    """Main-run ranges: [min best, max best] widened by range_margin_decades on each side, per hyperparameter."""
    tunable = [r for r in results if not r.untunable]
    lrs = [r.best_lr for r in tunable]
    wds = [r.best_wd for r in tunable if r.regularization == "wd"]
    if not lrs or not wds:
        raise ValueError(f"freeze_ranges: no tunable calibration recipes for lr ({len(lrs)}) or wd ({len(wds)})")
    m = cfg.tuning.range_margin_decades

    def widen(values: list[float]) -> tuple[float, float]:
        return (10 ** (math.log10(min(values)) - m), 10 ** (math.log10(max(values)) + m))

    ranges = Ranges(widen(lrs), widen(wds))
    report = {"best_lr": lrs, "best_wd": wds,
              "untunable_recipes": [r.recipe_id for r in results if r.untunable],
              "margin_decades": m}
    return ranges, report


def record_optuna(storage_path: str, study_name: str, points: list[dict[str, float]],
                  objectives: list[float | None], regularization: str, ranges: Ranges) -> None:
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    from optuna.trial import TrialState, create_trial

    # The default symlink lock needs admin rights on Windows.
    storage = JournalStorage(JournalFileBackend(storage_path, lock_obj=JournalFileOpenLock(storage_path)))
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")
    dists = {"lr": FloatDistribution(*ranges.lr, log=True)}
    if regularization == "wd":
        dists["wd"] = FloatDistribution(*ranges.wd, log=True)
    for i, (p, o) in enumerate(zip(points, objectives)):
        params = {k: p[k] for k in dists}
        attrs = {"trial_index": i}
        if o is None:
            study.add_trial(create_trial(params=params, distributions=dists, state=TrialState.FAIL,
                                         user_attrs=attrs))
        else:
            study.add_trial(create_trial(params=params, distributions=dists, value=o, user_attrs=attrs))
