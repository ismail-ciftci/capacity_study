"""Timeline and window selection. Membership is decided by target date: lo < target_date <= hi."""
from dataclasses import dataclass

import pandas as pd
import torch

from capacity_study.config import StudyConfig
from capacity_study.data.bank import Bank
from capacity_study.universe import ALL_TICKERS, TEST_ASSET, TIER_ASSETS


@dataclass(frozen=True)
class Timeline:
    history_start: pd.Timestamp
    tuning_cutoff: pd.Timestamp
    origin: pd.Timestamp
    data_end: pd.Timestamp

    def as_dict(self) -> dict[str, str]:
        return {k: str(v.date()) for k, v in vars(self).items()}


def timeline(cfg: StudyConfig, history_years: int) -> Timeline:
    end = pd.Timestamp(cfg.data.data_end)
    origin = end - pd.DateOffset(years=cfg.data.test_years)
    start = origin - pd.DateOffset(years=history_years)
    cutoff = (start + (origin - start) * cfg.tuning.tuning_fraction).normalize()
    return Timeline(start, cutoff, origin, end)


def _day(ts: pd.Timestamp) -> int:
    return int((ts.normalize() - pd.Timestamp("1970-01-01")).days)


def select(bank: Bank, tickers: tuple[str, ...], lo: pd.Timestamp, hi: pd.Timestamp) -> torch.Tensor:
    """Flat indices (asset * T + t) of valid windows of `tickers` with lo < target_date <= hi, row-major order."""
    asset_mask = torch.zeros(len(ALL_TICKERS), dtype=torch.bool, device=bank.valid.device)
    asset_mask[[ALL_TICKERS.index(t) for t in tickers]] = True
    date_mask = (bank.target_day > _day(lo)) & (bank.target_day <= _day(hi))
    mask = bank.valid & asset_mask[:, None] & date_mask[None, :]
    return torch.nonzero(mask.reshape(-1)).squeeze(1)


@dataclass(frozen=True)
class JobData:
    """Flat bank indices for one job. `train` is pooled over the tier; the others are test-asset only."""
    train: torch.Tensor
    train_side: torch.Tensor
    eval: torch.Tensor
    timeline: Timeline


def job_data(bank: Bank, cfg: StudyConfig, phase: str, tier: int, history: int) -> JobData:
    tl = timeline(cfg, history)
    assets, test = TIER_ASSETS[tier], (TEST_ASSET,)
    if phase in ("calibration", "tuning"):
        train_hi, eval_lo, eval_hi = tl.tuning_cutoff, tl.tuning_cutoff, tl.origin
    elif phase == "final":
        train_hi, eval_lo, eval_hi = tl.origin, tl.origin, tl.data_end
    else:
        raise ValueError(f"job_data: unknown phase {phase!r}")
    return JobData(train=select(bank, assets, tl.history_start, train_hi),
                   train_side=select(bank, test, tl.history_start, train_hi),
                   eval=select(bank, test, eval_lo, eval_hi),
                   timeline=tl)
