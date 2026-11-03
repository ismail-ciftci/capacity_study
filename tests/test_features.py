import numpy as np
import pandas as pd

from capacity_study.data.bank import build_bank_arrays
from capacity_study.data.features import haar_details, n_features, window_features, window_statistics
from capacity_study.universe import ALL_TICKERS


def _walks(n=50, length=16, seed=0):
    rng = np.random.default_rng(seed)
    return 4.6 + np.cumsum(0.02 * rng.standard_normal((n, length)), axis=1)


def test_dwt_preserves_energy():
    p = _walks()
    d, a = haar_details(p)
    np.testing.assert_allclose((d ** 2).sum(axis=1) + (a ** 2).sum(axis=1), (p ** 2).sum(axis=1), rtol=1e-12)


def test_layout_matches_brief():
    p = _walks(n=1)
    mu, sigma = p.mean(), p.std()
    q = (p[0] - mu) / sigma
    f = window_features(p)[0]
    assert f.shape == (n_features(16),) == (19,)
    np.testing.assert_allclose(f[:8], (q[0::2] - q[1::2]) / np.sqrt(2))   # finest details first, normalized window
    np.testing.assert_allclose(f[15:17], [q[-4:].std(), q[-8:].std()])     # volatility of the normalized prices
    np.testing.assert_allclose(f[17:], [mu, np.log(sigma)])                # re-injected statistics


def test_normalized_details_are_raw_details_over_sigma():
    p = _walks()
    _, sigma = window_statistics(p)
    raw, _ = haar_details(p)
    np.testing.assert_allclose(window_features(p)[:, :15], raw / sigma[:, None], rtol=1e-10)


def test_only_mu_and_log_sigma_see_level_and_scale():
    p = _walks()
    base = window_features(p)
    shifted = window_features(p + 1.7)
    np.testing.assert_allclose(shifted[:, :17], base[:, :17], atol=1e-9)
    np.testing.assert_allclose(shifted[:, 17], base[:, 17] + 1.7)
    mu = p.mean(axis=1, keepdims=True)
    scaled = window_features(mu + 3.0 * (p - mu))
    np.testing.assert_allclose(scaled[:, :17], base[:, :17], atol=1e-9)
    np.testing.assert_allclose(scaled[:, 18], base[:, 18] + np.log(3.0))


def test_vol_features_equal_detail_energy_identity():
    """pstd of the normalized prices over the last m days = sqrt(sum of squared details in that block / m)."""
    f = window_features(_walks())
    d1, d2, d3 = f[:, 0:8], f[:, 8:12], f[:, 12:14]
    blocks = {4: [d1[:, -2:], d2[:, -1:]], 8: [d1[:, -4:], d2[:, -2:], d3[:, -1:]]}
    for col, m in zip((15, 16), (4, 8)):
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
    for key in ("X", "scale", "offset"):
        np.testing.assert_array_equal(base[key][:, : t + 1], after[key][:, : t + 1])  # inputs up to t unchanged
    np.testing.assert_array_equal(base["y"][:, :t], after["y"][:, :t])                 # targets before t unchanged
    assert not np.array_equal(base["y"][:, t], after["y"][:, t])


def test_persistence_output_implies_zero_return():
    arrays = build_bank_arrays(_prices(), 16)
    a, t = 3, 50
    log_p = np.log(_prices()[ALL_TICKERS[a]].to_numpy())
    window = log_p[t - 15: t + 1]
    mu, sigma = window.mean(), window.std()
    np.testing.assert_allclose(arrays["scale"][a, t], sigma, rtol=1e-6)
    np.testing.assert_allclose(arrays["offset"][a, t], mu - window[-1], atol=1e-7)
    persistence_output = (window[-1] - mu) / sigma
    implied = arrays["scale"][a, t] * persistence_output + arrays["offset"][a, t]
    assert abs(implied) < 1e-6
    np.testing.assert_allclose(arrays["y"][a, t], log_p[t + 1] - log_p[t], rtol=1e-6)


def test_bank_drops_windows_with_missing_days_or_zero_sigma():
    prices = _prices()
    prices.iloc[30, 0] = np.nan
    prices.iloc[55:71, 1] = 50.0          # 16 equal closes: sigma = 0 for the window ending at 70
    arrays = build_bank_arrays(prices, 16)
    valid = arrays["valid"][0]
    assert not valid[29]                  # target missing
    assert not valid[30:46].any()         # window contains the missing day
    assert valid[46] and valid[28]
    assert not valid[:15].any()           # incomplete lookback
    assert not arrays["valid"][1][70] and arrays["valid"][1][69] and arrays["valid"][1][71]
