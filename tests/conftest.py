import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capacity_study.config import StudyConfig, config_from_dict
from capacity_study.data.fetch import sha256_file
from capacity_study.universe import ALL_TICKERS


def tiny_config_dict(tmp: Path) -> dict:
    return {
        "study": {"root_seed": 7},
        "data": {"raw_prices": str(tmp / "data/raw/prices.parquet"), "manifest": str(tmp / "data/raw/manifest.json"),
                 "bank_dir": str(tmp / "cache"), "data_end": "2024-12-31", "test_years": 1, "lookback": 16,
                 "price_adjustment": "split_dividend"},
        "grid": {"depths": [1, 2], "widths": [8], "regularizations": ["none", "wd"], "tiers": [1, 2],
                 "histories": [1, 2]},
        "training": {"epochs": 6, "steps_per_epoch": 3, "warmup_fraction": 0.2, "beta": 0.9, "beta2": 0.999,
                     "eps": 1e-8, "leaky_slope": 0.01, "checkpoint_every": 2, "pred_snapshot_every": 2,
                     "eval_chunk": 64},
        "tuning": {"n_trials": 3, "halton_seed": 0, "objective_tail": 2, "tuning_fraction": 0.8,
                   "calibration_lr": [1e-5, 1e-1], "calibration_wd": [1e-5, 1.0], "range_margin_decades": 0.75,
                   "calibration_cells": [[1, 1], [2, 2]], "calibration_corners": [[1, 8], [2, 8]]},
        "seeds": {"n_seeds": 3},
        "analysis": {"alpha": 0.05, "beta": 0.2, "v1_block": 2, "sentinel_test_r2": -1.0e12,
                     "sentinel_gap": 1.0e12, "sentinel_t1_logratio": 1.0e12, "sentinel_v1": 1.0e12},
    }


def write_synthetic_prices(tmp: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-06-01", "2024-12-31")
    data = 50 * np.exp(np.cumsum(0.015 * rng.standard_normal((len(idx), len(ALL_TICKERS))), axis=0))
    prices = pd.DataFrame(data, index=idx, columns=list(ALL_TICKERS))
    raw = tmp / "data/raw/prices.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(raw)
    (tmp / "data/raw/manifest.json").write_text(json.dumps(
        {"sha256": sha256_file(raw), "price_adjustment": "split_dividend", "tickers": list(ALL_TICKERS)}))


@pytest.fixture
def tiny(tmp_path: Path) -> tuple[StudyConfig, Path]:
    write_synthetic_prices(tmp_path)
    table = tiny_config_dict(tmp_path)
    cfg_path = tmp_path / "tiny.toml"
    cfg_path.write_text(_to_toml(table))
    return config_from_dict(table, "tiny"), cfg_path


def _to_toml(table: dict) -> str:
    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return repr(v)

    out = []
    for section, fields in table.items():
        out.append(f"[{section}]")
        out += [f"{k} = {value(v)}" for k, v in fields.items()]
    return "\n".join(out) + "\n"
