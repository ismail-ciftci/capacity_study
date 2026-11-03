"""Study configuration loaded from TOML.

Every result-affecting value lives in the TOML file. Fields have no defaults: a missing or unknown key raises
ConfigError. The resolved config is hashed (canonical JSON, sha256) to bind a study directory to it.
"""
import dataclasses
import hashlib
import json
import tomllib
import types
import typing
from dataclasses import dataclass
from pathlib import Path

from capacity_study.errors import ConfigError


@dataclass(frozen=True)
class StudySection:
    root_seed: int


@dataclass(frozen=True)
class DataSection:
    raw_prices: str
    manifest: str
    bank_dir: str
    data_end: str
    test_years: int
    lookback: int
    price_adjustment: str


@dataclass(frozen=True)
class GridSection:
    depths: tuple[int, ...]
    widths: tuple[int, ...]
    regularizations: tuple[str, ...]
    tiers: tuple[int, ...]
    histories: tuple[int, ...]


@dataclass(frozen=True)
class TrainingSection:
    epochs: int
    steps_per_epoch: int
    warmup_fraction: float
    beta: float
    beta2: float
    eps: float
    leaky_slope: float
    checkpoint_every: int
    pred_snapshot_every: int
    eval_chunk: int


@dataclass(frozen=True)
class TuningSection:
    n_trials: int
    seeds_per_trial: int
    halton_seed: int
    objective_tail: int
    tuning_fraction: float
    calibration_lr: tuple[float, float]
    calibration_wd: tuple[float, float]
    range_margin_decades: float
    calibration_cells: tuple[tuple[int, int], ...]
    calibration_corners: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SeedsSection:
    n_seeds: int


@dataclass(frozen=True)
class AnalysisSection:
    alpha: float
    beta: float
    v1_block: int
    sentinel_test_r2: float
    sentinel_gap: float
    sentinel_t1_logratio: float
    sentinel_v1: float


@dataclass(frozen=True)
class StudyConfig:
    study: StudySection
    data: DataSection
    grid: GridSection
    training: TrainingSection
    tuning: TuningSection
    seeds: SeedsSection
    analysis: AnalysisSection

    @property
    def warmup_steps(self) -> int:
        return int(round(self.training.warmup_fraction * self.training.epochs * self.training.steps_per_epoch))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


_ALLOWED_REGULARIZATIONS = ("none", "wd")
_ALLOWED_ADJUSTMENTS = ("split_dividend",)


def _coerce(value: object, tp: object, source: str, key: str) -> object:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(source, key, f"expected a table, got {type(value).__name__}")
        return _build(tp, value, source, key)
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, list):
            raise ConfigError(source, key, f"expected an array, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            if len(value) == 0:
                raise ConfigError(source, key, "array must not be empty")
            return tuple(_coerce(v, args[0], source, f"{key}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(source, key, f"expected {len(args)} elements, got {len(value)}")
        return tuple(_coerce(v, a, source, f"{key}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(source, key, f"expected bool, got {type(value).__name__}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(source, key, f"expected int, got {type(value).__name__}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(source, key, f"expected float, got {type(value).__name__}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(source, key, f"expected str, got {type(value).__name__}")
        return value
    if isinstance(tp, types.UnionType):
        raise ConfigError(source, key, "union types are not supported in config")
    raise ConfigError(source, key, f"unsupported field type {tp!r}")


def _build(cls: type, table: dict, source: str, prefix: str) -> object:
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(table) - set(names))
    if unknown:
        raise ConfigError(source, f"{prefix}.{unknown[0]}" if prefix else unknown[0], "unknown key")
    kwargs = {}
    for name in names:
        key = f"{prefix}.{name}" if prefix else name
        if name not in table:
            raise ConfigError(source, key, "missing (no defaults for result-affecting values)")
        kwargs[name] = _coerce(table[name], hints[name], source, key)
    return cls(**kwargs)


def _validate(cfg: StudyConfig, source: str) -> None:
    t, tu, a, g = cfg.training, cfg.tuning, cfg.analysis, cfg.grid
    for reg in g.regularizations:
        if reg not in _ALLOWED_REGULARIZATIONS:
            raise ConfigError(source, "grid.regularizations", f"unknown option {reg!r}")
    if cfg.data.price_adjustment not in _ALLOWED_ADJUSTMENTS:
        raise ConfigError(source, "data.price_adjustment", f"must be one of {_ALLOWED_ADJUSTMENTS}")
    if cfg.warmup_steps < 1:
        raise ConfigError(source, "training.warmup_fraction", "yields fewer than 1 warmup step")
    if tu.objective_tail > t.epochs:
        raise ConfigError(source, "tuning.objective_tail", "exceeds training.epochs")
    if 2 * a.v1_block > t.epochs:
        raise ConfigError(source, "analysis.v1_block", "two blocks exceed training.epochs")
    if not 0.0 < tu.tuning_fraction < 1.0:
        raise ConfigError(source, "tuning.tuning_fraction", "must be in (0, 1)")
    for name, (lo, hi) in (("calibration_lr", tu.calibration_lr), ("calibration_wd", tu.calibration_wd)):
        if not 0.0 < lo < hi:
            raise ConfigError(source, f"tuning.{name}", "must satisfy 0 < low < high")
    for tier, hist in tu.calibration_cells:
        if tier not in g.tiers or hist not in g.histories:
            raise ConfigError(source, "tuning.calibration_cells", f"cell ({tier}, {hist}) not in grid")
    for depth, width in tu.calibration_corners:
        if depth not in g.depths or width not in g.widths:
            raise ConfigError(source, "tuning.calibration_corners", f"corner ({depth}, {width}) not in grid")
    if t.checkpoint_every < 1 or t.pred_snapshot_every < 1 or t.eval_chunk < 1:
        raise ConfigError(source, "training", "checkpoint_every, pred_snapshot_every, eval_chunk must be >= 1")
    if tu.n_trials < 1 or tu.seeds_per_trial < 1 or cfg.seeds.n_seeds < 2:
        raise ConfigError(source, "tuning.n_trials/tuning.seeds_per_trial/seeds.n_seeds",
                          "need >= 1 trial, >= 1 seed per trial and >= 2 final seeds")


def config_from_dict(table: dict, source: str) -> StudyConfig:
    cfg = _build(StudyConfig, table, source, "")
    assert isinstance(cfg, StudyConfig)
    _validate(cfg, source)
    return cfg


def load_config(path: Path) -> StudyConfig:
    with open(path, "rb") as f:
        table = tomllib.load(f)
    return config_from_dict(table, str(path))
