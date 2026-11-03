"""Throughput and memory benchmark used to choose workers per device and to check the memory estimates.

Runs representative job shapes (cheapest, median, most expensive, and the largest memory estimate among the main
tuning packs) for a few epochs in W concurrent processes. Results go to <study>/bench/bench.json; scratch job
outputs stay under <study>/bench/ and never touch the study's jobs.
"""
import dataclasses
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch

from capacity_study.data.bank import load_bank
from capacity_study.data.splits import job_data
from capacity_study.orchestrate import estimate_memory_mb, load_study, members_per_recipe
from capacity_study.recipes import JobSpec, grid, pack_jobs
from capacity_study.store import StudyPaths, atomic_write_json, code_state, runtime_info, utc_now
from capacity_study.train import JobContext, run_job, tuning_members
from capacity_study.tuning import calibration_ranges


def _shapes(root: Path) -> list[dict]:
    cfg, _, record = load_study(root)
    bank = load_bank(cfg, record["bank"]["key"], record["bank"]["sha256"])
    rows = []
    for spec in pack_jobs(cfg, "tuning"):
        n = int(job_data(bank, cfg, "calibration", spec.tier, spec.history).train.numel())
        k = members_per_recipe(cfg, "tuning") * len(spec.recipe_ids)
        params = 19 * spec.width + spec.width + (spec.depth - 1) * (spec.width ** 2 + spec.width) + spec.width + 1
        rows.append({"tier": spec.tier, "history": spec.history, "depth": spec.depth, "width": spec.width,
                     "n_train": n, "members": k, "cost": k * n * params,
                     "est_mem_mb": estimate_memory_mb(cfg, spec, k, n)})
    rows.sort(key=lambda r: r["cost"])
    picks = {"cheapest": rows[0], "median": rows[len(rows) // 2], "costliest": rows[-1],
             "largest_memory": max(rows, key=lambda r: r["est_mem_mb"])}
    return [{"label": label} | row for label, row in picks.items()]


def bench_one(root: Path, device_str: str, tier: int, history: int, depth: int, width: int, epochs: int) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg, config_hash, record = load_study(root)
    cfg = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, epochs=epochs,
                                                                checkpoint_every=epochs,
                                                                pred_snapshot_every=epochs),
                              tuning=dataclasses.replace(cfg.tuning, objective_tail=min(epochs,
                                                                                        cfg.tuning.objective_tail)))
    device = torch.device(device_str)
    bank = load_bank(cfg, record["bank"]["key"], record["bank"]["sha256"]).to(device)
    index = {(r.depth, r.width, r.regularization, r.tier, r.history): r for r in grid(cfg)}
    ids = tuple(index[(depth, width, reg, tier, history)].recipe_id for reg in cfg.grid.regularizations)
    spec = JobSpec(f"bench-t{tier}-h{history}-d{depth}-w{width}", "calibration", tier, history, depth, width, ids)
    ranges = calibration_ranges(cfg)
    members = tuning_members(cfg, spec, {r.recipe_id: r for r in grid(cfg)}, ranges)
    scratch = StudyPaths(root / "bench" / uuid.uuid4().hex[:8])
    stamps: list[float] = []

    def stamp() -> None:
        torch.cuda.synchronize(device)  # bench only: the training loop itself never syncs per epoch
        stamps.append(time.perf_counter())

    ctx = JobContext(cfg, config_hash, scratch, spec, members, {"bench": True}, bank, device,
                     code_state(True, "bench"), runtime_info(device), False, lambda: False,
                     stamp, ranges)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    run_job(ctx)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - t0
    gaps = sorted(b - a for a, b in zip(stamps[1:], stamps[2:]))  # skip the first (warm-up) epoch
    if not gaps:
        raise ValueError("bench_one: need at least 3 epochs to measure a steady-state epoch")
    return {"seconds": seconds, "steady_epoch_seconds": gaps[len(gaps) // 2], "epochs": epochs, "steps": epochs * cfg.training.steps_per_epoch,
            "members": len(members), "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2 ** 20}


def bench(root: Path, device_str: str, epochs: int, worker_counts: list[int]) -> str:
    shapes = _shapes(root)
    results = []
    lines = [f"{'shape':<15}{'W':>3}{'job s/epoch':>13}{'member-steps/s':>16}{'peak MB':>9}{'est MB':>8}"]
    for shape in shapes:
        for w in worker_counts:
            cmd = [sys.executable, "-c",
                   "import json,sys; from pathlib import Path; from capacity_study.bench import bench_one; "
                   "print('BENCH ' + json.dumps(bench_one(Path(sys.argv[1]), sys.argv[2], *map(int, sys.argv[3:]))))",
                   str(root), device_str, str(shape["tier"]), str(shape["history"]), str(shape["depth"]),
                   str(shape["width"]), str(epochs)]
            procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) for _ in range(w)]
            runs = []
            for p in procs:
                out, _ = p.communicate()
                if p.returncode != 0:
                    raise RuntimeError(f"bench run failed for {shape['label']} W={w} (exit {p.returncode})")
                runs.append(json.loads(next(l for l in out.splitlines() if l.startswith("BENCH "))[6:]))
            slowest = max(r["steady_epoch_seconds"] for r in runs)
            steps_per_epoch = runs[0]["steps"] // runs[0]["epochs"]
            throughput = sum(r["members"] for r in runs) * steps_per_epoch / slowest
            peak = max(r["peak_reserved_mb"] for r in runs)
            results.append({"shape": shape, "workers": w, "runs": runs, "member_steps_per_s": throughput})
            lines.append(f"{shape['label']:<15}{w:>3}{slowest:>13.3f}{throughput:>16.0f}{peak:>9.0f}"
                         f"{shape['est_mem_mb']:>8.0f}")
    atomic_write_json(root / "bench" / "bench.json", {"utc": utc_now(), "device": device_str, "epochs": epochs,
                                                      "runtime": runtime_info(torch.device(device_str)),
                                                      "results": results})
    return "\n".join(lines)
