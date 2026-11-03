"""Feature bank: every asset x every window-end date, built once by `prepare` and loaded once per worker.

Layout (A assets in universe order, T = n_dates - 1 window-end positions; position t ends at date t and targets t+1):
  X           (A, T, 19) float32   window features, brief S2 (zeros where invalid)
  y           (A, T)     float32   realized return log P(t+1) - log P(t) (zeros where invalid)
  scale       (A, T)     float32   window sigma (ones where invalid)
  offset      (A, T)     float32   window mu - log P(t), computed in float64 (zeros where invalid)
  valid       (A, T)     bool      full lookback and target present (no missing day) and sigma > 0

The network predicts in window units; its implied return is scale * output + offset, which equals the forecast
mu + sigma * output minus log P(t). Losses and R^2 are computed on the implied return.
  end_day     (T,)       int32     last input date, days since 1970-01-01
  target_day  (T,)       int32     target date, days since 1970-01-01
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from capacity_study.config import StudyConfig
from capacity_study.data.features import n_features, window_features, window_statistics
from capacity_study.data.fetch import load_prices, sha256_file
from capacity_study.errors import DataIntegrityError, OverwriteRefusedError
from capacity_study.universe import ALL_TICKERS

FEATURE_VERSION = "window-normalized-haar-v2"


@dataclass
class Bank:
    X: torch.Tensor
    y: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor
    valid: torch.Tensor
    end_day: torch.Tensor
    target_day: torch.Tensor
    tickers: tuple[str, ...]
    key: str
    sha256: str

    @property
    def n_dates(self) -> int:
        return int(self.end_day.shape[0])

    def to(self, device: torch.device | str) -> "Bank":
        return Bank(self.X.to(device), self.y.to(device), self.scale.to(device), self.offset.to(device),
                    self.valid.to(device), self.end_day.to(device), self.target_day.to(device), self.tickers,
                    self.key, self.sha256)


def bank_key(cfg: StudyConfig, manifest_sha: str) -> str:
    spec = {"manifest_sha256": manifest_sha, "lookback": cfg.data.lookback, "features": FEATURE_VERSION,
            "tickers": list(ALL_TICKERS)}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def bank_paths(cfg: StudyConfig, key: str) -> tuple[Path, Path]:
    base = Path(cfg.data.bank_dir)
    return base / f"bank_{key}.pt", base / f"bank_{key}.json"


def _days(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.values.astype("datetime64[D]").astype(np.int64)).astype(np.int32)


def build_bank_arrays(prices: pd.DataFrame, lookback: int) -> dict[str, np.ndarray]:
    prices = prices.reindex(columns=list(ALL_TICKERS))
    log_p = np.log(prices.to_numpy(dtype=np.float64))  # (n_dates, A); NaN where missing
    n_dates, n_assets = log_p.shape
    T = n_dates - 1
    X = np.zeros((n_assets, T, n_features(lookback)), dtype=np.float32)
    y = np.zeros((n_assets, T), dtype=np.float32)
    scale = np.ones((n_assets, T), dtype=np.float32)
    offset = np.zeros((n_assets, T), dtype=np.float32)
    valid = np.zeros((n_assets, T), dtype=bool)
    for a in range(n_assets):
        windows = sliding_window_view(log_p[:, a], lookback)[:-1]  # ends at t = lookback-1 .. n_dates-2
        ends = np.arange(lookback - 1, n_dates - 1)
        target = log_p[ends + 1, a] - log_p[ends, a]
        complete = np.isfinite(windows).all(axis=1) & np.isfinite(target)
        mu, sigma = window_statistics(np.where(complete[:, None], windows, 0.0))
        ok = complete & (sigma > 0)  # a window of equal closes cannot be normalized (brief S2: dropped)
        w = windows[ok]
        X[a, ends[ok]] = window_features(w).astype(np.float32)
        y[a, ends[ok]] = target[ok].astype(np.float32)
        scale[a, ends[ok]] = sigma[ok].astype(np.float32)
        offset[a, ends[ok]] = (mu[ok] - w[:, -1]).astype(np.float32)
        valid[a, ends[ok]] = True
    days = _days(pd.DatetimeIndex(prices.index))
    return {"X": X, "y": y, "scale": scale, "offset": offset, "valid": valid, "end_day": days[:-1],
            "target_day": days[1:]}


def prepare_bank(cfg: StudyConfig) -> Path:
    prices, manifest = load_prices(cfg)
    key = bank_key(cfg, manifest["sha256"])
    pt_path, meta_path = bank_paths(cfg, key)
    if pt_path.exists() or meta_path.exists():
        raise OverwriteRefusedError(str(pt_path), "prepare")
    arrays = build_bank_arrays(prices, cfg.data.lookback)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pt_path.with_suffix(".pt.tmp")
    torch.save({k: torch.from_numpy(v) for k, v in arrays.items()} | {"tickers": list(ALL_TICKERS)}, tmp)
    tmp.replace(pt_path)
    meta = {"key": key, "sha256": sha256_file(pt_path), "manifest_sha256": manifest["sha256"],
            "features": FEATURE_VERSION, "lookback": cfg.data.lookback, "tickers": list(ALL_TICKERS),
            "valid_windows_per_ticker": dict(zip(ALL_TICKERS, arrays["valid"].sum(axis=1).tolist()))}
    meta_path.write_text(json.dumps(meta, indent=2))
    return pt_path


def load_bank(cfg: StudyConfig, key: str, expected_sha256: str) -> Bank:
    pt_path, _ = bank_paths(cfg, key)
    actual = sha256_file(pt_path)
    if actual != expected_sha256:
        raise DataIntegrityError(str(pt_path), expected_sha256, actual)
    blob = torch.load(pt_path, map_location="cpu", weights_only=True)
    tickers = tuple(blob["tickers"])
    if tickers != ALL_TICKERS:
        raise DataIntegrityError(str(pt_path), ",".join(ALL_TICKERS), ",".join(tickers))
    return Bank(blob["X"], blob["y"], blob["scale"], blob["offset"], blob["valid"], blob["end_day"],
                blob["target_day"], tickers, key, actual)
