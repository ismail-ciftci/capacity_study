import pandas as pd
import pytest
import torch

from capacity_study.config import config_from_dict
from capacity_study.data.bank import Bank, build_bank_arrays
from capacity_study.data.splits import job_data, timeline
from capacity_study.train import batch_bounds
from capacity_study.universe import ALL_TICKERS, SECTOR_PICKS, TEST_ASSET, TIER_ASSETS

from conftest import tiny_config_dict


def test_tiers():
    sizes = {t: len(a) for t, a in TIER_ASSETS.items()}
    assert sizes == {1: 1, 2: 4, 3: 12, 4: 24, 5: 24}
    for t in (1, 2, 3):
        assert set(TIER_ASSETS[t]) < set(TIER_ASSETS[t + 1])
    assert TIER_ASSETS[5][:4] == TIER_ASSETS[2]
    assert all(len(p) == 4 for p in SECTOR_PICKS.values()) and len(SECTOR_PICKS) == 5
    assert len(ALL_TICKERS) == 44 and len(set(ALL_TICKERS)) == 44
    assert all(TIER_ASSETS[t][0] == TEST_ASSET for t in TIER_ASSETS)


def _bank(tmp_path):
    from conftest import write_synthetic_prices
    write_synthetic_prices(tmp_path)
    prices = pd.read_parquet(tmp_path / "data/raw/prices.parquet")
    arrays = build_bank_arrays(prices, 16)
    return Bank(*(torch.from_numpy(arrays[k]) for k in ("X", "y", "valid", "end_day", "target_day")),
                ALL_TICKERS, "k", "s")


@pytest.mark.parametrize("phase", ["tuning", "final"])
def test_no_label_crosses_a_cutoff(tmp_path, phase):
    cfg = config_from_dict(tiny_config_dict(tmp_path), "t")
    bank = _bank(tmp_path)
    T = bank.n_dates
    for tier in (1, 2):
        for history in (1, 2):
            d = job_data(bank, cfg, phase, tier, history)
            tl = timeline(cfg, history)
            hi = tl.tuning_cutoff if phase == "tuning" else tl.origin
            day = lambda ts: (ts - pd.Timestamp("1970-01-01")).days
            for name, idx, lo_ts, hi_ts in (("train", d.train, tl.history_start, hi),
                                            ("eval", d.eval, hi, tl.origin if phase == "tuning" else tl.data_end)):
                tgt = bank.target_day[idx % T]
                assert (tgt > day(lo_ts)).all() and (tgt <= day(hi_ts)).all(), name
            assert set((d.train_side // T).tolist()) == {0} and set((d.eval // T).tolist()) == {0}
            assets = set((d.train // T).tolist())
            assert assets == {ALL_TICKERS.index(t) for t in TIER_ASSETS[tier]}
            assert set(d.train_side.tolist()) <= set(d.train.tolist())


def test_timeline_fractions(tmp_path):
    cfg = config_from_dict(tiny_config_dict(tmp_path), "t")
    tl = timeline(cfg, 2)
    assert tl.origin == pd.Timestamp("2023-12-31") and tl.history_start == pd.Timestamp("2021-12-31")
    frac = (tl.tuning_cutoff - tl.history_start) / (tl.origin - tl.history_start)
    assert abs(frac - 0.8) < 0.01


@pytest.mark.parametrize("n", [25, 26, 49, 750, 108_013])
def test_batch_bounds(n):
    bounds = batch_bounds(n, 25)
    sizes = [hi - lo for lo, hi in bounds]
    assert len(bounds) == 25 and sum(sizes) == n and max(sizes) - min(sizes) <= 1
    assert bounds[0][0] == 0 and all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
