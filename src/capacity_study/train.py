"""Runs one job: a batched model of K members trained for the configured epochs, with per-epoch evaluation at the
evaluation weights x, chunked metric logging, atomic checkpoints, pause/resume and a DONE marker written last.

GPU residency: the feature bank is already on the device; the job gathers its datasets once (GPU -> GPU), draws
permutations on the GPU, and only moves data to the host at flushes (metrics, prediction snapshots, checkpoints).

Output mapping (brief S2): the network predicts in window units; the implied return is scale * output + offset
(the forecast mu + sigma * output minus log P(t)). The loss, R^2 and prediction snapshots use the implied return;
prediction spread (brief S9) uses the raw output.
"""
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from capacity_study import rng
from capacity_study.config import StudyConfig
from capacity_study.data.bank import Bank
from capacity_study.data.splits import job_data
from capacity_study.errors import CheckpointMismatchError, JobStateError
from capacity_study.model import BatchedMLP
from capacity_study.optim import BatchedAdamWScheduleFree
from capacity_study.recipes import JobSpec, Recipe
from capacity_study.store import (JobPaths, StudyPaths, append_jsonl, atomic_torch_save, atomic_write_bytes,
                                  atomic_write_json, sha256_path, utc_now)
from capacity_study.tuning import (Ranges, TuningResult, objective, record_optuna, select_best, trial_objectives,
                                   trial_points)

METRIC_COLUMNS = ("train_loss_y", "pooled_train_mse_x", "r2_train_x", "r2_eval_x", "output_std_eval_x",
                  "x_norm", "x_dist_init", "finite")


@dataclass(frozen=True)
class Member:
    index: int
    recipe_id: int
    regularization: str
    role: str            # "trial" | "seed"
    role_index: int      # trial index (tuning) or seed index (final)
    seed: int            # seed index: within a trial (tuning) or of the recipe (final)
    lr: float
    wd: float
    stream_key: tuple[int, int, int]


def tuning_members(cfg: StudyConfig, spec: JobSpec, recipes: dict[int, Recipe], ranges: Ranges) -> list[Member]:
    """Trial-major: member (trial t, seed j). Seed j's stream is shared by every trial of the recipe (paired)."""
    members = []
    for rid in spec.recipe_ids:
        reg = recipes[rid].regularization
        for t, p in enumerate(trial_points(cfg, reg, ranges)):
            for j in range(cfg.tuning.seeds_per_trial):
                members.append(Member(len(members), rid, reg, "trial", t, j, p["lr"], p["wd"],
                                      rng.member_key(spec.phase, rid, j)))
    return members


def final_members(cfg: StudyConfig, spec: JobSpec, recipes: dict[int, Recipe],
                  tuned: dict[int, TuningResult]) -> list[Member]:
    members = []
    for rid in spec.recipe_ids:
        result = tuned[rid]
        if result.untunable:
            continue
        for s in range(cfg.seeds.n_seeds):
            members.append(Member(len(members), rid, recipes[rid].regularization, "seed", s, s, result.best_lr,
                                  result.best_wd, rng.member_key("final", rid, s)))
    return members


@dataclass
class JobContext:
    cfg: StudyConfig
    config_hash: str
    study: StudyPaths
    spec: JobSpec
    members: list[Member]
    job_meta: dict                       # extra provenance for job.json (ranges / tuning sources, data hashes)
    bank: Bank                           # resident on `device`
    device: torch.device
    code: dict                           # store.code_state()
    runtime: dict                        # store.runtime_info()
    allow_code_change: bool
    should_pause: Callable[[], bool]
    heartbeat: Callable[[], None]
    ranges: Ranges | None = None


def batch_bounds(n: int, steps: int) -> list[tuple[int, int]]:
    """`steps` contiguous slices of near-equal size (differ by at most 1; the first n % steps get the extra)."""
    if n < steps:
        raise ValueError(f"batch_bounds: {n} windows cannot fill {steps} batches")
    base, extra = divmod(n, steps)
    bounds, start = [], 0
    for j in range(steps):
        size = base + (1 if j < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def implied_return(output: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """output (K, n) in window units; scale, offset (n,) -> implied return (K, n) = forecast - log P(t)."""
    return output * scale.unsqueeze(0) + offset.unsqueeze(0)


def r2_vs_persistence(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """pred (K, n), y (n,) -> (K,) float64: 1 - sum((y - pred)^2) / sum(y^2)."""
    yd = y.double()
    return 1.0 - ((yd.unsqueeze(0) - pred.double()) ** 2).sum(dim=1) / (yd ** 2).sum()


@torch.no_grad()
def predict(model: BatchedMLP, X: torch.Tensor, chunk: int) -> torch.Tensor:
    k = model.num_members
    parts = [model(X[i:i + chunk].unsqueeze(0).expand(k, -1, -1)) for i in range(0, X.shape[0], chunk)]
    return torch.cat(parts, dim=1)


@torch.no_grad()
def pooled_mse(model: BatchedMLP, X: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor,
               chunk: int) -> torch.Tensor:
    """MSE of the implied return over a whole set, per member (float64)."""
    k = model.num_members
    total = torch.zeros(k, dtype=torch.float64, device=X.device)
    for i in range(0, X.shape[0], chunk):
        out = model(X[i:i + chunk].unsqueeze(0).expand(k, -1, -1))
        pred = implied_return(out, scale[i:i + chunk], offset[i:i + chunk])
        total += ((pred.double() - y[i:i + chunk].double().unsqueeze(0)) ** 2).sum(dim=1)
    return total / X.shape[0]


def _member_table(members: list[Member]) -> list[dict]:
    return [asdict(m) | {"stream_key": list(m.stream_key)} for m in members]


def _write_or_verify_job_json(ctx: JobContext, paths: JobPaths, data_sizes: dict) -> None:
    record = {"spec": ctx.spec.to_dict(), "config_hash": ctx.config_hash, "config": ctx.cfg.to_dict(),
              "members": _member_table(ctx.members), "data": data_sizes,
              "bank": {"key": ctx.bank.key, "sha256": ctx.bank.sha256}} | ctx.job_meta
    if paths.job_json.exists():
        existing = json.loads(paths.job_json.read_text())
        for field in ("spec", "config_hash", "members", "data", "bank"):
            if json.loads(json.dumps(record[field], default=str)) != existing[field]:
                raise CheckpointMismatchError(ctx.spec.job_id, f"job.json:{field}", existing[field], record[field])
        return
    atomic_write_json(paths.job_json, record | {"created_utc": utc_now()})


def _event(paths: JobPaths, ctx: JobContext, kind: str, **extra) -> None:
    code = {k: v for k, v in ctx.code.items() if k != "git_diff"}
    append_jsonl(paths.events, {"event": kind, "utc": utc_now(), "job_id": ctx.spec.job_id} | code
                 | {"runtime": ctx.runtime} | extra)


def _clean_after(paths: JobPaths, epoch: int) -> None:
    """Remove outputs written after the checkpoint epoch (they will be recomputed identically)."""
    for p in list(paths.metrics.glob("e*.parquet")):
        if int(p.stem.split("-")[1]) > epoch:
            p.unlink()
    for p in list(paths.preds.glob("e*.pt")):
        if int(p.stem[1:]) > epoch:
            p.unlink()


def read_metrics(paths: JobPaths, epochs: int) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(paths.metrics.glob("e*.parquet"))]
    df = pd.concat(frames, ignore_index=True).sort_values(["epoch", "member"], kind="stable")
    counts = df.groupby("epoch").size()
    if list(counts.index) != list(range(1, epochs + 1)) or counts.nunique() != 1:
        raise JobStateError(str(paths.root), "read_metrics", "metric chunks do not cover each epoch exactly once")
    return df.reset_index(drop=True)


def run_job(ctx: JobContext) -> str:
    """Returns 'done' or 'paused'."""
    cfg, spec, device = ctx.cfg, ctx.spec, ctx.device
    tr = cfg.training
    paths = ctx.study.job(spec.job_id)
    if paths.done.exists():
        raise JobStateError(spec.job_id, "run_job", "already complete (DONE.json exists)")
    paths.root.mkdir(parents=True, exist_ok=True)

    data = job_data(ctx.bank, cfg, spec.phase, spec.tier, spec.history)
    Xf, yf = ctx.bank.X.reshape(-1, ctx.bank.X.shape[-1]), ctx.bank.y.reshape(-1)
    Sf, Of = ctx.bank.scale.reshape(-1), ctx.bank.offset.reshape(-1)

    def gather(idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(a[idx].contiguous() for a in (Xf, yf, Sf, Of))

    X_train, y_train, S_train, O_train = gather(data.train)
    X_side, y_side, S_side, O_side = gather(data.train_side)
    X_eval, y_eval, S_eval, O_eval = gather(data.eval)
    n = X_train.shape[0]
    data_sizes = {"n_train": n, "n_train_side": X_side.shape[0], "n_eval": X_eval.shape[0],
                  "timeline": data.timeline.as_dict(), "steps_per_epoch": tr.steps_per_epoch,
                  "batch_size_min": n // tr.steps_per_epoch, "batch_size_max": -(-n // tr.steps_per_epoch)}
    _write_or_verify_job_json(ctx, paths, data_sizes)

    if not ctx.members:
        _event(paths, ctx, "done_empty", reason="every recipe in this pack is untunable")
        _finish(ctx, paths, summary={"empty": True})
        return "done"

    k = len(ctx.members)
    model = BatchedMLP(k, X_train.shape[1], spec.depth, spec.width, tr.leaky_slope)
    model.initialize([rng.init_seed(cfg.study.root_seed, m.stream_key) for m in ctx.members])
    if not paths.weights_init.exists():
        atomic_torch_save(paths.weights_init, {n_: p.detach().clone() for n_, p in model.named_parameters()})
    model.to(device)
    init_weights = [w.detach().clone() for w in model.weights]
    opt = BatchedAdamWScheduleFree(model.ordered_parameters(),
                                   torch.tensor([m.lr for m in ctx.members], dtype=torch.float32, device=device),
                                   torch.tensor([m.wd for m in ctx.members], dtype=torch.float32, device=device),
                                   tr.beta, tr.beta2, tr.eps, cfg.warmup_steps)

    start_epoch = 0
    if paths.ckpt.exists():
        ckpt = torch.load(paths.ckpt, map_location="cpu", weights_only=True)
        for field, current in (("job_id", spec.job_id), ("config_hash", ctx.config_hash)):
            if ckpt[field] != current:
                raise CheckpointMismatchError(spec.job_id, field, ckpt[field], current)
        if ckpt["git_commit"] != ctx.code["git_commit"] and not ctx.allow_code_change:
            raise CheckpointMismatchError(spec.job_id, "git_commit", ckpt["git_commit"], ctx.code["git_commit"])
        with torch.no_grad():
            for (p, _), saved in zip(model.ordered_parameters(), ckpt["params"]):
                p.copy_(saved)
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        _clean_after(paths, start_epoch)
        _event(paths, ctx, "resume", from_epoch=start_epoch,
               code_changed=ckpt["git_commit"] != ctx.code["git_commit"])
    else:
        _clean_after(paths, 0)
        _event(paths, ctx, "start")

    # Permutation streams: members sharing a stream key share the data order.
    keys = sorted({m.stream_key for m in ctx.members})
    member_stream = torch.tensor([keys.index(m.stream_key) for m in ctx.members], device=device)
    gen = torch.Generator(device=device)
    bounds = batch_bounds(n, tr.steps_per_epoch)
    is_final = spec.phase == "final"

    buffer = torch.zeros(tr.checkpoint_every + 1, k, len(METRIC_COLUMNS), dtype=torch.float64, device=device)
    sched_log: list[tuple[float, float]] = []
    pending_preds: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    chunk_start, chunk_t0 = start_epoch + 1, time.perf_counter()

    def flush(epoch: int) -> None:
        nonlocal chunk_start, chunk_t0, sched_log, pending_preds
        rows = buffer[: epoch - chunk_start + 1].cpu().numpy()
        epochs = np.arange(chunk_start, epoch + 1)
        frame = pd.DataFrame({"epoch": np.repeat(epochs, k), "member": np.tile(np.arange(k), len(epochs)),
                              "recipe_id": np.tile([m.recipe_id for m in ctx.members], len(epochs))})
        for c, name in enumerate(METRIC_COLUMNS):
            frame[name] = rows[:, :, c].reshape(-1)
        frame["sched"] = np.repeat([s for s, _ in sched_log], k)
        frame["ckp1"] = np.repeat([c for _, c in sched_log], k)
        paths.metrics.mkdir(parents=True, exist_ok=True)
        tmp = paths.metrics / f"e{chunk_start:04d}-{epoch:04d}.parquet.tmp"
        frame.to_parquet(tmp, index=False)
        tmp.replace(paths.metrics / f"e{chunk_start:04d}-{epoch:04d}.parquet")
        for e, p_side, p_eval, o_eval in pending_preds:
            atomic_torch_save(paths.preds / f"e{e:04d}.pt", {"train_side": p_side.cpu(), "eval": p_eval.cpu(),
                                                             "eval_output": o_eval.cpu()})
        params = [p.detach().to("cpu", copy=True) for p, _ in model.ordered_parameters()]
        atomic_torch_save(paths.ckpt, {"job_id": spec.job_id, "config_hash": ctx.config_hash,
                                       "git_commit": ctx.code["git_commit"], "epoch": epoch,
                                       "params": params, "optimizer": opt.state_dict()})
        _event(paths, ctx, "checkpoint", epoch=epoch, chunk_seconds=time.perf_counter() - chunk_t0)
        chunk_start, chunk_t0, sched_log, pending_preds = epoch + 1, time.perf_counter(), [], []

    for epoch in range(start_epoch + 1, tr.epochs + 1):
        opt.train()
        perms = []
        for key in keys:
            gen.manual_seed(rng.order_seed(cfg.study.root_seed, key, epoch))
            perms.append(torch.randperm(n, generator=gen, device=device))
        member_perm = torch.stack(perms).index_select(0, member_stream)
        loss_acc = torch.zeros(k, device=device)
        for lo, hi in bounds:
            idx = member_perm[:, lo:hi]
            pred = model(X_train[idx]) * S_train[idx] + O_train[idx]   # implied return, (K, b)
            loss = ((pred - y_train[idx]) ** 2).mean(dim=1)
            loss.sum().backward()
            opt.step()
            opt.zero_grad()
            loss_acc += loss.detach()
        sched_log.append((opt.last_sched, opt.last_ckp1))

        opt.eval()
        p_side = implied_return(predict(model, X_side, tr.eval_chunk), S_side, O_side)
        o_eval = predict(model, X_eval, tr.eval_chunk)
        p_eval = implied_return(o_eval, S_eval, O_eval)
        row = buffer[epoch - chunk_start]
        row[:, 0] = loss_acc.double() / len(bounds)
        row[:, 1] = pooled_mse(model, X_train, y_train, S_train, O_train, tr.eval_chunk)
        row[:, 2] = r2_vs_persistence(p_side, y_side)
        row[:, 3] = r2_vs_persistence(p_eval, y_eval)
        row[:, 4] = o_eval.double().std(dim=1, correction=1)
        sq_norm = sum((w.detach().double() ** 2).flatten(1).sum(1) for w in model.weights)
        sq_dist = sum(((w.detach().double() - w0.double()) ** 2).flatten(1).sum(1)
                      for w, w0 in zip(model.weights, init_weights))
        row[:, 5], row[:, 6] = sq_norm.sqrt(), sq_dist.sqrt()
        finite = torch.isfinite(loss_acc)
        for p, _ in model.ordered_parameters():
            finite &= torch.isfinite(p.detach()).flatten(1).all(dim=1)
        row[:, 7] = finite.double()
        snapshot = (is_final and epoch % tr.pred_snapshot_every == 0) or epoch == tr.epochs
        if snapshot:
            pending_preds.append((epoch, p_side.clone(), p_eval.clone(), o_eval.clone()))
        opt.train()

        ctx.heartbeat()
        pause = ctx.should_pause() and epoch < tr.epochs
        if epoch % tr.checkpoint_every == 0 or epoch == tr.epochs or pause:
            flush(epoch)
        if pause:
            _event(paths, ctx, "pause", epoch=epoch)
            return "paused"

    opt.eval()
    atomic_torch_save(paths.weights_final, {n_: p.detach().cpu() for n_, p in model.named_parameters()})
    summary = {"members": k}
    if spec.phase in ("calibration", "tuning"):
        summary["tuning"] = _write_tuning_results(ctx, paths)
    _finish(ctx, paths, summary)
    return "done"


def _write_tuning_results(ctx: JobContext, paths: JobPaths) -> dict:
    cfg = ctx.cfg
    df = read_metrics(paths, cfg.training.epochs)
    tail = df[df["epoch"] > cfg.training.epochs - cfg.tuning.objective_tail]
    finite_all = df.groupby("member")["finite"].min().to_numpy() == 1.0
    r2_tail = tail.pivot(index="epoch", columns="member", values="r2_eval_x").to_numpy()
    member_objectives = objective(r2_tail, finite_all)
    out = {}
    for rid in ctx.spec.recipe_ids:
        mine = [m for m in ctx.members if m.recipe_id == rid]
        reg = mine[0].regularization
        n_trials = max(m.role_index for m in mine) + 1
        per_trial = [[member_objectives[m.index] for m in mine if m.role_index == t] for t in range(n_trials)]
        points = [next({"lr": m.lr, "wd": m.wd} for m in mine if m.role_index == t) for t in range(n_trials)]
        result = select_best(rid, reg, points, trial_objectives(per_trial), ctx.ranges)
        atomic_write_json(paths.tuning / f"recipe_{rid}.json", result.to_dict() | {
            "ranges": ctx.ranges.to_dict(), "seed_objectives": per_trial})
        journal = paths.tuning / f"optuna_recipe_{rid}.log"
        if journal.exists():
            journal.unlink()  # a previous attempt died mid-write; the journal is derived from metrics
        record_optuna(str(journal), f"recipe_{rid}", points, result.objectives, reg, ctx.ranges)
        out[str(rid)] = {"untunable": result.untunable, "best_lr": result.best_lr, "best_wd": result.best_wd,
                         "edge_hit": result.edge_hit, "infeasible_fraction": result.infeasible_fraction}
    return out


def _finish(ctx: JobContext, paths: JobPaths, summary: dict) -> None:
    outputs = {str(p.relative_to(paths.root)).replace("\\", "/"): sha256_path(p) for p in paths.outputs()}
    record = {"job_id": ctx.spec.job_id, "config_hash": ctx.config_hash, "completed_utc": utc_now(),
              "git_commit": ctx.code["git_commit"], "git_dirty": ctx.code["git_dirty"],
              "summary": summary, "outputs_sha256": outputs}
    _event(paths, ctx, "done")
    atomic_write_bytes(paths.done, json.dumps(record, indent=2, sort_keys=True, default=str).encode())
    if paths.ckpt.exists():
        paths.ckpt.unlink()


def load_tuning_results(study: StudyPaths, tuning_job_id: str, recipe_ids: tuple[int, ...]) -> tuple[
        dict[int, TuningResult], str]:
    paths = study.job(tuning_job_id)
    if not paths.done.exists():
        raise JobStateError(tuning_job_id, "load_tuning_results", "tuning job is not complete")
    results = {}
    for rid in recipe_ids:
        d = json.loads((paths.tuning / f"recipe_{rid}.json").read_text())
        results[rid] = TuningResult(d["recipe_id"], d["regularization"], d["objectives"], d["best_trial"],
                                    d["best_lr"], d["best_wd"], d["edge_hit"], d["infeasible_fraction"])
    return results, sha256_path(paths.done)
