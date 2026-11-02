"""Window features: orthonormal Haar DWT of 16 log prices plus three volatility features (19 inputs).

Follows the guide's pseudocode exactly: details from the finest to the coarsest level, each (a - b)/sqrt(2) over
consecutive pairs (a, b); then the final approximation; then the population std of the last 4, 8 and 16 log prices.
Computed in float64.
"""
import numpy as np

VOL_SPANS = (4, 8, 16)


def haar_features(log_prices: np.ndarray) -> np.ndarray:
    """log_prices: (n, L) float64, oldest -> newest, L a power of two. Returns (n, L + 3) float64."""
    p = np.asarray(log_prices, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"haar_features: expected (n, L) array, got shape {p.shape}")
    length = p.shape[1]
    if length < 2 or length & (length - 1):
        raise ValueError(f"haar_features: window length {length} is not a power of two >= 2")
    if max(VOL_SPANS) > length:
        raise ValueError(f"haar_features: window length {length} shorter than volatility span {max(VOL_SPANS)}")

    a = p
    details = []
    while a.shape[1] > 1:
        first, second = a[:, 0::2], a[:, 1::2]
        details.append((first - second) / np.sqrt(2.0))
        a = (first + second) / np.sqrt(2.0)
    vols = [p[:, -m:].std(axis=1, ddof=0)[:, None] for m in VOL_SPANS]
    return np.concatenate([*details, a, *vols], axis=1)
