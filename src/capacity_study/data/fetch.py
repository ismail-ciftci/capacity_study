"""One-time price download. Writes the raw snapshot and its sha256 manifest; never overwrites."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_study.config import StudyConfig
from capacity_study.errors import DataCoverageError, DataIntegrityError, OverwriteRefusedError
from capacity_study.universe import ALL_TICKERS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def required_start(cfg: StudyConfig) -> pd.Timestamp:
    """Earliest date whose price is needed: longest history start minus the lookback (with slack for holidays)."""
    end = pd.Timestamp(cfg.data.data_end)
    origin = end - pd.DateOffset(years=cfg.data.test_years)
    start = origin - pd.DateOffset(years=max(cfg.grid.histories))
    return start - pd.Timedelta(days=2 * cfg.data.lookback + 7)


def check_coverage(prices: pd.DataFrame, cfg: StudyConfig, operation: str) -> None:
    missing = [t for t in ALL_TICKERS if t not in prices.columns]
    if missing:
        raise DataCoverageError(operation, missing, "ticker absent from price table")
    start, end = required_start(cfg), pd.Timestamp(cfg.data.data_end)
    late = [t for t in ALL_TICKERS if prices[t].first_valid_index() is None or prices[t].first_valid_index() > start]
    if late:
        raise DataCoverageError(operation, late, f"no price on or before {start.date()}")
    early_end = [t for t in ALL_TICKERS if prices[t].last_valid_index() < end]
    if early_end:
        raise DataCoverageError(operation, early_end, f"prices end before data_end {end.date()}")


def fetch_prices(cfg: StudyConfig) -> Path:
    """Download adjusted closes for every ticker, verify coverage, write parquet + manifest."""
    import yfinance as yf

    raw_path, manifest_path = Path(cfg.data.raw_prices), Path(cfg.data.manifest)
    for path in (raw_path, manifest_path):
        if path.exists():
            raise OverwriteRefusedError(str(path), "fetch-data")

    start = required_start(cfg) - pd.Timedelta(days=30)
    end_exclusive = pd.Timestamp(cfg.data.data_end) + pd.Timedelta(days=1)
    frame = yf.download(list(ALL_TICKERS), start=str(start.date()), end=str(end_exclusive.date()),
                        auto_adjust=True, progress=True, threads=True)
    closes = frame["Close"].reindex(columns=list(ALL_TICKERS)).astype(np.float64)
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    closes = closes.sort_index()
    check_coverage(closes, cfg, "fetch-data")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(raw_path)
    manifest = {
        "file": raw_path.name,
        "sha256": sha256_file(raw_path),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
        "yfinance_version": yf.__version__,
        "price_adjustment": cfg.data.price_adjustment,
        "auto_adjust": True,
        "tickers": list(ALL_TICKERS),
        "first_date": str(closes.index[0].date()),
        "last_date": str(closes.index[-1].date()),
        "n_dates": int(len(closes)),
        "missing_per_ticker": {t: int(closes[t].isna().sum()) for t in ALL_TICKERS},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return raw_path


def load_prices(cfg: StudyConfig) -> tuple[pd.DataFrame, dict]:
    """Load the raw snapshot after verifying it against its manifest."""
    raw_path, manifest_path = Path(cfg.data.raw_prices), Path(cfg.data.manifest)
    manifest = json.loads(manifest_path.read_text())
    actual = sha256_file(raw_path)
    if actual != manifest["sha256"]:
        raise DataIntegrityError(str(raw_path), manifest["sha256"], actual)
    if manifest["price_adjustment"] != cfg.data.price_adjustment:
        raise DataIntegrityError(str(manifest_path), cfg.data.price_adjustment, manifest["price_adjustment"])
    prices = pd.read_parquet(raw_path)
    check_coverage(prices, cfg, "load-prices")
    return prices, manifest
