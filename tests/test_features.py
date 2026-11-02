import numpy as np
import pandas as pd

from capacity_study.data.bank import build_bank_arrays
from capacity_study.data.features import haar_features
from capacity_study.universe import ALL_TICKERS


def _walks(n=50, length=16, seed=0):
    rng = np.random.default_rng(seed)
    return 4.6 + np.cumsum(0.02 * rng.standard_normal((n, length)), axis=1)


def test_dwt_preserves_energy():
    p = _walks()
    f = haar_features(p)
    np.testing.assert_allclose((f[:, :16] ** 2).sum(axis=1), (p ** 2).sum(axis=1), rtol=1e-12)


def test_layout_matches_guide_pseudocode():
    p = np.arange(16, dtype=np.float64)[None, :]
    f = haar_features(p)[0]
    np.testing.assert_allclose(f[:8], (p[0, 0::2] - p[0, 1::2]) / np.sqrt(2))  # finest details first
    np.testing.assert_allclose(f[15], p.sum() / 4.0)                             # approximation = sum / 4
    np.testing.assert_allclose(f[16:], [p[0, -m:].std() for m in (4, 8, 16)])


def test_vol_features_equal_detail_energy_identity():
    """pstd over the last m days = sqrt(sum of squared details in that block / m) (guide Ch.2)."""
    p = _walks()
    f = haar_features(p)
    d1, d2, d3, d4 = f[:, 0:8], f[:, 8:12], f[:, 12:14], f[:, 14:15]
    blocks = {4: [d1[:, -2:], d2[:, -1:]],
              8: [d1[:, -4:], d2[:, -2:], d3[:, -1:]],
              16: [d1, d2, d3, d4]}
    for col, m in zip((16, 17, 18), (4, 8, 16)):
        energy = sum((b ** 2).sum(axis=1) for b in blocks[m])
        np.testing.assert_allclose(f[:, col], np.sqrt(energy / m), rtol=1e-10)


def _prices(n_dates=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates)
    data = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((n_dates, len(ALL_TICKERS))), axis=0))
    return pd.DataFrame(data, index=idx, columns=list(ALL_TICKERS))


def test_bank_is_causal():
    prices = _prices()
    base = build_bank_arrays(prices, 16)
    t = 40
    perturbed = prices.copy()
    perturbed.iloc[t + 1:] *= 1.5
    after = build_bank_arrays(perturbed, 16)
    np.testing.assert_array_equal(base["X"][:, : t + 1], after["X"][:, : t + 1])  # inputs up to t unchanged
    np.testing.assert_array_equal(base["y"][:, :t], after["y"][:, :t])             # targets before t unchanged
    assert not np.array_equal(base["y"][:, t], after["y"][:, t])


def test_bank_drops_windows_with_missing_days():
    prices = _prices()
    prices.iloc[30, 0] = np.nan
    arrays = build_bank_arrays(prices, 16)
    valid = arrays["valid"][0]
    assert not valid[29]                  # target missing
    assert not valid[30:46].any()         # window contains the missing day
    assert valid[46] and valid[28]
    assert not valid[:15].any()           # incomplete lookback
