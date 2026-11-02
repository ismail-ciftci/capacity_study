"""Regionalization (guide Ch. 12): contiguity-constrained agglomerative merging by equivalence.

Start with every recipe as a region; repeatedly merge the adjacent pair with the smallest D (exact ties broken by
the pair's smallest recipe IDs); stop when the smallest D >= 1. Merged regions take the union of their members'
neighbours. Untunable recipes (sentinels) merge only with each other (D = 0) and never with tunable ones.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class Member:
    recipe_id: int
    value: float        # seed mean (mean rule) or T1 log ratio (log-ratio rule)
    sd: float           # seed sd (mean rule) or jackknife SE (log-ratio rule)
    n: int              # seeds used
    untunable: bool


@dataclass
class Region:
    members: list[Member] = field(default_factory=list)

    @property
    def ids(self) -> list[int]:
        return sorted(m.recipe_id for m in self.members)

    @property
    def untunable(self) -> bool:
        return all(m.untunable for m in self.members)

    @property
    def mu(self) -> float:
        return float(np.mean([m.value for m in self.members]))

    @property
    def sigma(self) -> float:
        return float(math.sqrt(np.mean([m.sd ** 2 for m in self.members])))

    @property
    def heterogeneity(self) -> float:
        vals = [m.value for m in self.members]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _mean_rule_D(a: Region, b: Region, delta_mu: float, alpha: float) -> float:
    terms = [(m.sd ** 2 / (m.n * len(r.members) ** 2), m.n - 1) for r in (a, b) for m in r.members]
    se2 = sum(c for c, _ in terms)
    sigma_p = math.sqrt((a.sigma ** 2 + b.sigma ** 2) / 2)
    if sigma_p == 0.0:
        return 0.0 if a.mu == b.mu else math.inf
    df = se2 ** 2 / sum(c ** 2 / d for c, d in terms) if se2 > 0 else math.inf
    t = stats.t.ppf(1 - alpha, df)
    return (abs(a.mu - b.mu) + t * math.sqrt(se2)) / (delta_mu * sigma_p)


def _logratio_rule_D(a: Region, b: Region, delta_sigma: float, alpha: float) -> float:
    se_a = math.sqrt(sum(m.sd ** 2 for m in a.members)) / len(a.members)
    se_b = math.sqrt(sum(m.sd ** 2 for m in b.members)) / len(b.members)
    se = math.sqrt(se_a ** 2 + se_b ** 2)
    return (abs(a.mu - b.mu) + stats.norm.ppf(1 - alpha) * se) / delta_sigma


def merge_statistic(rule: str, a: Region, b: Region, delta: float, alpha: float) -> float:
    if a.untunable or b.untunable:
        return 0.0 if (a.untunable and b.untunable) else math.inf
    if rule == "mean":
        return _mean_rule_D(a, b, delta, alpha)
    if rule == "logratio":
        return _logratio_rule_D(a, b, delta, alpha)
    raise ValueError(f"merge_statistic: unknown rule {rule!r}")


@dataclass
class Regionalization:
    labels: dict[int, int]            # recipe_id -> region ID (1..R, ordered by smallest member recipe ID)
    regions: dict[int, Region]        # region ID -> region
    merges: list[dict]                # merge history


def regionalize(members: list[Member], adjacency: dict[int, list[int]], rule: str, delta: float,
                alpha: float) -> Regionalization:
    regions = {m.recipe_id: Region([m]) for m in members}  # keyed by a provisional id
    neigh = {rid: set(adjacency[rid]) for rid in regions}
    merges = []

    def key(a: int, b: int) -> tuple:
        return (merge_statistic(rule, regions[a], regions[b], delta, alpha),
                min(regions[a].ids[0], regions[b].ids[0]), max(regions[a].ids[0], regions[b].ids[0]))

    pairs = {(a, b): key(a, b) for a in neigh for b in neigh[a] if a < b}
    while pairs:
        (a, b), k = min(pairs.items(), key=lambda item: item[1])
        if not k[0] < 1.0:
            break
        regions[a].members += regions.pop(b).members
        merges.append({"step": len(merges) + 1, "kept": a, "absorbed": b, "D": k[0], "size": len(regions[a].members)})
        neigh[a] = (neigh[a] | neigh.pop(b)) - {a, b}
        for other in neigh[a]:
            neigh[other].discard(b)
            neigh[other].add(a)
        pairs = {p: v for p, v in pairs.items() if a not in p and b not in p}
        for other in neigh[a]:
            pairs[(min(a, other), max(a, other))] = key(min(a, other), max(a, other))
    ordered = sorted(regions.values(), key=lambda r: r.ids[0])
    final = {i + 1: r for i, r in enumerate(ordered)}
    labels = {rid: region_id for region_id, r in final.items() for rid in r.ids}
    return Regionalization(labels, final, merges)


def label_components(labels: dict[int, str], adjacency: dict[int, list[int]]) -> dict[int, int]:
    """Status map: connected areas of identical labels. Region IDs ordered by smallest recipe ID."""
    seen, comps = set(), []
    for start in sorted(labels):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            r = stack.pop()
            comp.append(r)
            for n in adjacency[r]:
                if n not in seen and labels[n] == labels[start]:
                    seen.add(n)
                    stack.append(n)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: c[0])
    return {rid: i + 1 for i, comp in enumerate(comps) for rid in comp}
