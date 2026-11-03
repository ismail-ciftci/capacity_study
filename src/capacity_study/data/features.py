"""Window features (brief S2): the 16 log prices are normalized by their own mean mu and population std sigma, the
normalized window is transformed by the orthonormal Haar DWT (15 details, finest level first, each (a - b)/sqrt(2)
over consecutive pairs (a, b); the approximation of a centred window is 0 and is dropped), followed by the population
std of the normalized prices over the last 4 and 8 days, then the re-injected statistics mu and log sigma: 19 inputs.

The network predicts the target in window units; the forecast is mu + sigma * output. Everything here is float64.
"""
import numpy as np

VOL_SPANS = (4, 8)


def n_features(lookback: int) -> int:
    """(lookback - 1) details + the volatility features + mu and log sigma; 19 for the 16-day lookback."""
    return lookback - 1 + len(VOL_SPANS) + 2


def haar_details(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x: (n, L), L a power of two. Returns (details (n, L-1), finest level first; approximation (n, 1))."""
    length = x.shape[1]
    if length < 2 or length & (length - 1):
        raise ValueError(f"haar_details: window length {length} is not a power of two >= 2")
    a, details = x, []
    while a.shape[1] > 1:
        first, second = a[:, 0::2], a[:, 1::2]
        details.append((first - second) / np.sqrt(2.0))
        a = (first + second) / np.sqrt(2.0)
    return np.concatenate(details, axis=1), a


def window_statistics(log_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std (ddof 0) of each window's log prices. log_prices: (n, L)."""
    p = np.asarray(log_prices, dtype=np.float64)
    return p.mean(axis=1), p.std(axis=1, ddof=0)


def window_features(log_prices: np.ndarray) -> np.ndarray:
    """log_prices: (n, L) float64, oldest -> newest, every window with sigma > 0. Returns (n, n_features(L))."""
    p = np.asarray(log_prices, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < max(VOL_SPANS):
        raise ValueError(f"window_features: expected (n, L >= {max(VOL_SPANS)}) array, got shape {p.shape}")
    mu, sigma = window_statistics(p)
    if not np.all(sigma > 0):
        raise ValueError(f"window_features: {int((sigma <= 0).sum())} windows have sigma = 0 (drop them first)")
    q = (p - mu[:, None]) / sigma[:, None]
    details, _ = haar_details(q)
    vols = [q[:, -m:].std(axis=1, ddof=0)[:, None] for m in VOL_SPANS]
    return np.concatenate([details, *vols, mu[:, None], np.log(sigma)[:, None]], axis=1)
