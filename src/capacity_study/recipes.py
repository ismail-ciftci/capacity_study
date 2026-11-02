"""Recipe grid, adjacency, and packing of recipes into jobs.

recipe_id = index in product(depth, width, regularization, tier, history), in that order. A job packs the
regularization options of one (phase, tier, history, depth, width): same data, batch size and shapes.
"""
import itertools
from dataclasses import dataclass

from capacity_study.config import StudyConfig

PHASES = ("calibration", "tuning", "final")


@dataclass(frozen=True)
class Recipe:
    recipe_id: int
    depth: int
    width: int
    regularization: str
    tier: int
    history: int


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    phase: str
    tier: int
    history: int
    depth: int
    width: int
    recipe_ids: tuple[int, ...]

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "phase": self.phase, "tier": self.tier, "history": self.history,
                "depth": self.depth, "width": self.width, "recipe_ids": list(self.recipe_ids)}

    @staticmethod
    def from_dict(d: dict) -> "JobSpec":
        return JobSpec(d["job_id"], d["phase"], d["tier"], d["history"], d["depth"], d["width"],
                       tuple(d["recipe_ids"]))


def grid(cfg: StudyConfig) -> list[Recipe]:
    g = cfg.grid
    combos = itertools.product(g.depths, g.widths, g.regularizations, g.tiers, g.histories)
    return [Recipe(i, *c) for i, c in enumerate(combos)]


def recipe_index(cfg: StudyConfig) -> dict[tuple, Recipe]:
    return {(r.depth, r.width, r.regularization, r.tier, r.history): r for r in grid(cfg)}


def job_id(phase: str, tier: int, history: int, depth: int, width: int) -> str:
    return f"{phase}-t{tier}-h{history}-d{depth}-w{width}"


def pack_jobs(cfg: StudyConfig, phase: str) -> list[JobSpec]:
    if phase not in PHASES:
        raise ValueError(f"pack_jobs: unknown phase {phase!r}")
    index = recipe_index(cfg)
    if phase == "calibration":
        cells = [(t, h, d, w) for t, h in cfg.tuning.calibration_cells for d, w in cfg.tuning.calibration_corners]
    else:
        cells = list(itertools.product(cfg.grid.tiers, cfg.grid.histories, cfg.grid.depths, cfg.grid.widths))
    jobs = []
    for tier, history, depth, width in cells:
        ids = tuple(index[(depth, width, reg, tier, history)].recipe_id for reg in cfg.grid.regularizations)
        jobs.append(JobSpec(job_id(phase, tier, history, depth, width), phase, tier, history, depth, width, ids))
    return jobs


def adjacency(cfg: StudyConfig) -> dict[int, list[int]]:
    """Recipes differing in one factor by one step (tiers chained 1-2-3-4-5; regularization: the other option)."""
    g = cfg.grid
    index = recipe_index(cfg)
    ordered = {"depth": g.depths, "width": g.widths, "tier": g.tiers, "history": g.histories}
    out: dict[int, list[int]] = {}
    for r in grid(cfg):
        neighbours = []
        for factor, levels in ordered.items():
            pos = levels.index(getattr(r, factor))
            for step in (-1, 1):
                if 0 <= pos + step < len(levels):
                    key = {"depth": r.depth, "width": r.width, "regularization": r.regularization,
                           "tier": r.tier, "history": r.history} | {factor: levels[pos + step]}
                    neighbours.append(index[(key["depth"], key["width"], key["regularization"], key["tier"],
                                             key["history"])].recipe_id)
        for reg in g.regularizations:
            if reg != r.regularization:
                neighbours.append(index[(r.depth, r.width, reg, r.tier, r.history)].recipe_id)
        out[r.recipe_id] = sorted(neighbours)
    return out
