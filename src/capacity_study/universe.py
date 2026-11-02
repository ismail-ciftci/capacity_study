"""The fixed asset draw (one-time decision, not an experiment parameter).

US equities with long S&P 500 tenure. Financial tiers 1-4 are nested; tier 5 is diverse at tier 4's size and its
four financials are tier 2's four. JPM is the test asset in every tier. Survivorship bias is accepted (C5).
Current Yahoo tickers are used: BNY (formerly BK, Bank of New York Mellon), MRSH (formerly MMC, Marsh & McLennan).
"""

TEST_ASSET = "JPM"

_TIER2_ADD = ("BAC", "WFC", "C")
_TIER3_ADD = ("GS", "MS", "USB", "PNC", "AXP", "MET", "AIG", "BNY")
_TIER4_ADD = ("SCHW", "COF", "PRU", "AFL", "ALL", "TRV", "CB", "MRSH", "PGR", "STT", "NTRS", "TROW")

SECTOR_PICKS = {
    "industrials": ("GE", "HON", "MMM", "CAT"),
    "health_care": ("JNJ", "PFE", "MRK", "ABT"),
    "information_technology": ("MSFT", "IBM", "INTC", "ORCL"),
    "consumer_discretionary": ("HD", "MCD", "NKE", "LOW"),
    "utilities": ("DUK", "SO", "AEP", "ED"),
}

TIER_ASSETS: dict[int, tuple[str, ...]] = {
    1: (TEST_ASSET,),
    2: (TEST_ASSET, *_TIER2_ADD),
    3: (TEST_ASSET, *_TIER2_ADD, *_TIER3_ADD),
    4: (TEST_ASSET, *_TIER2_ADD, *_TIER3_ADD, *_TIER4_ADD),
    5: (TEST_ASSET, *_TIER2_ADD, *(t for picks in SECTOR_PICKS.values() for t in picks)),
}

# Every ticker in the study, in a fixed order (index = asset index in the feature bank).
ALL_TICKERS: tuple[str, ...] = tuple(dict.fromkeys(t for tier in sorted(TIER_ASSETS) for t in TIER_ASSETS[tier]))
