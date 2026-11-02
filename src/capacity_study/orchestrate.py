"""Study lifecycle: init, enqueue, freeze ranges, worker loop, launcher, status."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from capacity_study.config import StudyConfig, config_from_dict, load_config
from capacity_study.data.bank import Bank, bank_key, bank_paths, load_bank
from capacity_study.data.fetch import sha256_file
from capacity_study.data.splits import job_data
from capacity_study.errors import ConfigError, DataIntegrityError, JobStateError, StudyStateError
from capacity_study.ledger import Ledger
from capacity_study.recipes import JobSpec, grid, pack_jobs
from capacity_study.store import (StudyPaths, append_jsonl, code_state, runtime_info, sha256_path, utc_now,
                                  write_json_once)
from capacity_study.train import JobContext, final_members, load_tuning_results, run_job, tuning_members
from capacity_study.tuning import Ranges, calibration_ranges, freeze_ranges

# ---------------------------------------------------------------------------------------------- study files


def load_study(root: Path) -> tuple[StudyConfig, str, dict]:
    paths = StudyPaths(root)
    if not paths.study_json.exists():
        raise StudyStateError(str(root), "load", "study.json not found; run init-study first")
    record = json.loads(paths.study_json.read_text())
    cfg = config_from_dict(record["config"], str(paths.study_json))
    if cfg.config_hash() != record["config_hash"]:
        raise ConfigError(str(paths.study_json), "config_hash", "stored hash does not match stored config")
    return cfg, record["config_hash"], record


def init_study(config_path: Path, root: Path, allow_dirty: bool) -> Path:
    cfg = load_config(config_path)
    paths = StudyPaths(root)
    if paths.study_json.exists():
        existing = json.loads(paths.study_json.read_text())
        if existing["config_hash"] != cfg.config_hash():
            raise StudyStateError(str(root), "init-study", "exists with a different config hash; use a new name")
        return paths.study_json
    manifest = json.loads(Path(cfg.data.manifest).read_text())
    key = bank_key(cfg, manifest["sha256"])
    pt_path, meta_path = bank_paths(cfg, key)
    if not pt_path.exists():
        raise StudyStateError(str(root), "init-study", f"feature bank {pt_path} missing; run prepare first")
    meta = json.loads(meta_path.read_text())
    actual = sha256_file(pt_path)
    if actual != meta["sha256"]:
        raise DataIntegrityError(str(pt_path), meta["sha256"], actual)
    root.mkdir(parents=True, exist_ok=True)
    record = {"config": cfg.to_dict(), "config_hash": cfg.config_hash(), "config_source": str(config_path),
              "bank": {"key": key, "sha256": actual, "manifest_sha256": manifest["sha256"]},
              "created_utc": utc_now(), "created_by": code_state(allow_dirty, "init-study")}
    write_json_once(paths.study_json, record, "init-study")
    Ledger(paths.ledger).close()
    return paths.study_json


# ---------------------------------------------------------------------------------------------- enqueue

def estimate_memory_mb(cfg: StudyConfig, spec: JobSpec, members: int, n_train: int) -> float:
    """Rough peak device memory of one job (excluding the per-process context and the resident bank)."""
    b_max = -(-n_train // cfg.training.steps_per_epoch)
    w, d = spec.width, spec.depth
    params = 19 * w + w + (d - 1) * (w * w + w) + w + 1
    act_train = members * b_max * w * 4 * (3 * d + 2)
    act_eval = members * cfg.training.eval_chunk * w * 4 * 3
    state = members * params * 4 * 7
    perms = members * n_train * 8 * 2
    data = n_train * 20 * 4 * 2
    return 1.5 * (act_train + act_eval + state + perms + data) / 2 ** 20 + 64.0


def _members_per_job(cfg: StudyConfig, phase: str, spec: JobSpec) -> int:
    per_recipe = cfg.tuning.n_trials if phase in ("calibration", "tuning") else cfg.seeds.n_seeds
    return per_recipe * len(spec.recipe_ids)


def enqueue(root: Path, phase: str) -> int:
    cfg, _, record = load_study(root)
    paths = StudyPaths(root)
    if phase == "tuning" and not paths.ranges.exists():
        raise StudyStateError(str(root), "enqueue tuning", "ranges.json missing; run freeze-ranges first")
    bank = load_bank(cfg, record["bank"]["key"], record["bank"]["sha256"])
    phases = ["tuning", "final"] if phase == "tuning" else [phase]
    entries = []
    for ph in phases:
        for spec in pack_jobs(cfg, ph):
            n_train = int(job_data(bank, cfg, ph, spec.tier, spec.history).train.numel())
            k = _members_per_job(cfg, ph, spec)
            params = 19 * spec.width + spec.width + (spec.depth - 1) * (spec.width ** 2 + spec.width) + spec.width + 1
            depends = spec.job_id.replace("final-", "tuning-", 1) if ph == "final" else None
            entries.append((spec, float(k * n_train * params), estimate_memory_mb(cfg, spec, k, n_train), depends))
    ledger = Ledger(paths.ledger)
    try:
        return ledger.add_jobs(entries)
    finally:
        ledger.close()


def freeze(root: Path) -> Path:
    cfg, _, _ = load_study(root)
    paths = StudyPaths(root)
    specs = pack_jobs(cfg, "calibration")
    missing = [s.job_id for s in specs if not paths.job(s.job_id).done.exists()]
    if missing:
        raise StudyStateError(str(root), "freeze-ranges", f"calibration jobs not complete: {missing}")
    results, sources = [], {}
    for spec in specs:
        res, done_sha = load_tuning_results(paths, spec.job_id, spec.recipe_ids)
        results += list(res.values())
        sources[spec.job_id] = done_sha
    ranges, report = freeze_ranges(cfg, results)
    write_json_once(paths.ranges, {"ranges": ranges.to_dict(), "report": report, "calibration_done_sha256": sources,
                                   "created_utc": utc_now()}, "freeze-ranges")
    return paths.ranges


# ---------------------------------------------------------------------------------------------- worker

class PauseSignal:
    def __init__(self, flag_path: Path):
        self.flag_path = flag_path
        self.requested = False
        for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None), signal.SIGTERM):
            if sig is not None:
                signal.signal(sig, self._handle)

    def _handle(self, signum, frame) -> None:
        self.requested = True

    def __call__(self) -> bool:
        return self.requested or self.flag_path.exists()


def _resolve(cfg: StudyConfig, paths: StudyPaths, spec: JobSpec) -> tuple[list, dict, Ranges | None]:
    recipes = {r.recipe_id: r for r in grid(cfg)}
    if spec.phase == "calibration":
        ranges = calibration_ranges(cfg)
        return tuning_members(cfg, spec, recipes, ranges), {"ranges": ranges.to_dict(), "ranges_source": "config"}, ranges
    if spec.phase == "tuning":
        ranges = Ranges.from_dict(json.loads(paths.ranges.read_text())["ranges"])
        meta = {"ranges": ranges.to_dict(), "ranges_source": "ranges.json",
                "ranges_sha256": sha256_path(paths.ranges)}
        return tuning_members(cfg, spec, recipes, ranges), meta, ranges
    tuning_job = spec.job_id.replace("final-", "tuning-", 1)
    tuned, done_sha = load_tuning_results(paths, tuning_job, spec.recipe_ids)
    meta = {"tuning_job": tuning_job, "tuning_done_sha256": done_sha,
            "untunable_recipes": [rid for rid, r in tuned.items() if r.untunable]}
    return final_members(cfg, spec, recipes, tuned), meta, None


def worker_main(root: Path, device_str: str, mem_budget_mb: float, allow_dirty: bool, allow_code_change: bool,
                max_consecutive_failures: int, idle_poll_seconds: float, heartbeat_seconds: float) -> int:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg, config_hash, record = load_study(root)
    paths = StudyPaths(root)
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, mem_budget_mb * 2 ** 20 / total), device)
    code = code_state(allow_dirty, "worker")
    runtime = runtime_info(device)
    bank: Bank = load_bank(cfg, record["bank"]["key"], record["bank"]["sha256"]).to(device)  # resident for life
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{device_str}"
    pause = PauseSignal(paths.pause_flag)
    ledger = Ledger(paths.ledger)
    failures = 0
    log = paths.root / "workers.jsonl"
    append_jsonl(log, {"event": "worker_start", "utc": utc_now(), "worker": worker_id,
                       "mem_budget_mb": mem_budget_mb} | {k: v for k, v in code.items() if k != "git_diff"})
    try:
        while not pause():
            claimed = ledger.claim(worker_id, device_str, mem_budget_mb)
            if claimed is None:
                remaining = sum(c for _, s, c in ledger.status_counts() if s in ("pending", "running"))
                if remaining == 0:
                    break
                time.sleep(idle_poll_seconds)
                continue
            spec = claimed.spec
            last_beat = [time.monotonic()]

            def beat(job_id: str = spec.job_id) -> None:
                if time.monotonic() - last_beat[0] >= heartbeat_seconds:
                    ledger.heartbeat(job_id, worker_id)
                    last_beat[0] = time.monotonic()

            try:
                members, meta, ranges = _resolve(cfg, paths, spec)
                ctx = JobContext(cfg, config_hash, paths, spec, members, meta, bank, device, code, runtime,
                                 allow_code_change, pause, beat, ranges)
                outcome = run_job(ctx)
            except JobStateError:
                raise
            except Exception as exc:
                failures += 1
                tb = traceback.format_exc()
                append_jsonl(paths.job(spec.job_id).events,
                             {"event": "failed", "utc": utc_now(), "worker": worker_id, "error": repr(exc),
                              "traceback": tb})
                ledger.fail(spec.job_id, worker_id, tb)
                if failures >= max_consecutive_failures:
                    append_jsonl(log, {"event": "worker_stop_failures", "utc": utc_now(), "worker": worker_id})
                    return 2
                continue
            if outcome == "paused":
                ledger.release(spec.job_id, worker_id)
                break
            ledger.complete(spec.job_id, worker_id)
            failures = 0
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        append_jsonl(log, {"event": "worker_exit", "utc": utc_now(), "worker": worker_id})
        ledger.close()
    return 0


# ---------------------------------------------------------------------------------------------- launcher

def default_mem_budget_mb(device_str: str, workers: int, context_mb: float = 450.0) -> float:
    device = torch.device(device_str)
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 20
    return 0.92 * total - workers * context_mb


def launch(root: Path, devices: list[str], workers_per_device: int, mem_budget_mb: float | None,
           allow_dirty: bool, allow_code_change: bool, stale_seconds: float) -> int:
    paths = StudyPaths(root)
    load_study(root)
    if paths.pause_flag.exists():
        raise StudyStateError(str(root), "launch", "PAUSE flag present; use `resume` to clear it and launch")
    ledger = Ledger(paths.ledger)
    try:
        fixed = ledger.reconcile(paths, stale_seconds)
    finally:
        ledger.close()
    print(f"reconcile: {fixed}")
    procs = []
    for dev in devices:
        budget = mem_budget_mb if mem_budget_mb is not None else default_mem_budget_mb(dev, workers_per_device)
        for _ in range(workers_per_device):
            cmd = [sys.executable, "-m", "capacity_study", "worker", "--study", str(root), "--device", dev,
                   "--mem-budget-mb", str(budget)]
            if allow_dirty:
                cmd.append("--allow-dirty")
            if allow_code_change:
                cmd.append("--allow-code-change")
            procs.append(subprocess.Popen(cmd))
    print(f"launched {len(procs)} workers on {devices}; `pause` (or Ctrl+C) checkpoints and stops them")
    codes = []
    for p in procs:
        while True:
            try:
                codes.append(p.wait())
                break
            except KeyboardInterrupt:
                continue  # workers received the same Ctrl+C and are checkpointing
    return max(codes) if codes else 0


def status(root: Path) -> str:
    paths = StudyPaths(root)
    ledger = Ledger(paths.ledger)
    try:
        lines = [f"study {root}" + ("  [PAUSE flag set]" if paths.pause_flag.exists() else "")]
        for phase, state, count in ledger.status_counts():
            lines.append(f"  {phase:<12} {state:<8} {count}")
        now = time.time()
        for job_id, device, worker, hb in ledger.running():
            lines.append(f"  running {job_id} on {device} ({worker}), heartbeat {now - hb:.0f}s ago")
        for job_id, error in ledger.failures():
            lines.append(f"  FAILED {job_id}: {error.strip().splitlines()[-1] if error else ''}")
    finally:
        ledger.close()
    commits = set()
    for done in (paths.root / "jobs").glob("*/DONE.json") if (paths.root / "jobs").exists() else []:
        commits.add(json.loads(done.read_text())["git_commit"])
    if len(commits) > 1:
        lines.append(f"  WARNING: completed jobs span {len(commits)} code commits: {sorted(map(str, commits))}")
    return "\n".join(lines)
