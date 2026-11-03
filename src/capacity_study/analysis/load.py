"""Load raw final-stage outputs per recipe. Requires every final job of the grid to be complete."""
import json
from dataclasses import dataclass

import numpy as np
import torch

from capacity_study.config import StudyConfig
from capacity_study.errors import StudyStateError
from capacity_study.recipes import Recipe, grid, pack_jobs
from capacity_study.store import StudyPaths, sha256_path
from capacity_study.train import load_tuning_results, read_metrics
from capacity_study.tuning import TuningResult


@dataclass
class RecipeRaw:
    recipe: Recipe
    tuning: TuningResult
    final_job: str
    n_train: int
    batch_size: float                 # mean final-stage batch size N / steps_per_epoch
    train_curve: np.ndarray | None    # (S, E) R2 on test-asset training windows, at x
    test_curve: np.ndarray | None     # (S, E) R2 on the test window, at x
    preds_train: np.ndarray | None    # (S, n_train_side) final-epoch predictions
    preds_test: np.ndarray | None     # (S, n_test)
    outputs_test: np.ndarray | None   # (S, n_test) final-epoch raw network output in window units (brief S9 spread)
    rel_weight_change: np.ndarray | None  # (S,) ||x - x1|| / ||x1|| over weight matrices
    finite: np.ndarray | None         # (S,) True if the seed stayed finite throughout

    @property
    def untunable(self) -> bool:
        return self.tuning.untunable


def _weight_norms(path, members: list[int]) -> list[torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return [blob[k][members].double() for k in sorted(blob) if k.startswith("weights.")]


def load_recipes(cfg: StudyConfig, paths: StudyPaths) -> tuple[list[RecipeRaw], dict[str, str]]:
    """Returns the raw data per recipe (canonical order) and the DONE.json sha256 of every input job."""
    specs = pack_jobs(cfg, "final")
    missing = [s.job_id for s in specs if not paths.job(s.job_id).done.exists()]
    if missing:
        raise StudyStateError(str(paths.root), "analyze", f"{len(missing)} final jobs incomplete, e.g. {missing[:3]}")
    recipes = {r.recipe_id: r for r in grid(cfg)}
    epochs = cfg.training.epochs
    out: dict[int, RecipeRaw] = {}
    sources: dict[str, str] = {}
    for spec in specs:
        tuning_job = spec.job_id.replace("final-", "tuning-", 1)
        tuned, tuning_sha = load_tuning_results(paths, tuning_job, spec.recipe_ids)
        jp = paths.job(spec.job_id)
        sources[tuning_job], sources[spec.job_id] = tuning_sha, sha256_path(jp.done)
        job = json.loads(jp.job_json.read_text())
        n_train = job["data"]["n_train"]
        b = n_train / cfg.training.steps_per_epoch
        members = job["members"]
        if not members:
            for rid in spec.recipe_ids:
                out[rid] = RecipeRaw(recipes[rid], tuned[rid], spec.job_id, n_train, b, *([None] * 7))
            continue
        metrics = read_metrics(jp, epochs)
        preds = torch.load(jp.preds / f"e{epochs:04d}.pt", map_location="cpu", weights_only=True)
        for rid in spec.recipe_ids:
            if tuned[rid].untunable:
                out[rid] = RecipeRaw(recipes[rid], tuned[rid], spec.job_id, n_train, b, *([None] * 7))
                continue
            idx = [m["index"] for m in members if m["recipe_id"] == rid]
            sub = metrics[metrics["member"].isin(idx)]
            curve = lambda col: sub.pivot(index="member", columns="epoch", values=col).loc[idx].to_numpy()
            finite = sub.groupby("member")["finite"].min().loc[idx].to_numpy() == 1.0
            w0 = _weight_norms(jp.weights_init, idx)
            w1 = _weight_norms(jp.weights_final, idx)
            num = torch.sqrt(sum(((a - b_) ** 2).flatten(1).sum(1) for a, b_ in zip(w1, w0)))
            den = torch.sqrt(sum((a ** 2).flatten(1).sum(1) for a in w0))
            out[rid] = RecipeRaw(recipes[rid], tuned[rid], spec.job_id, n_train, b,
                                 curve("r2_train_x"), curve("r2_eval_x"),
                                 preds["train_side"][idx].double().numpy(), preds["eval"][idx].double().numpy(),
                                 preds["eval_output"][idx].double().numpy(), (num / den).numpy(), finite)
    return [out[rid] for rid in sorted(out)], sources
