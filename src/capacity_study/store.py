"""Run-directory layout, atomic writes, provenance capture.

runs/<study>/
  study.json          resolved config, config hash, bank key + sha256, creation provenance
  ledger.sqlite       job queue (scheduling index; the filesystem DONE markers are the truth)
  ranges.json         frozen main-run tuning ranges (written once)
  PAUSE               present = workers checkpoint and exit at the next epoch end
  jobs/<job_id>/      job.json, events.jsonl, metrics/, preds/, weights_*.pt, ckpt.pt, tuning/, DONE.json
"""
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from capacity_study.errors import DirtyTreeError, OverwriteRefusedError, ProvenanceError

REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, default=str).encode())


def write_json_once(path: Path, obj: object, operation: str) -> None:
    if path.exists():
        raise OverwriteRefusedError(str(path), operation)
    atomic_write_json(path, obj)


def atomic_torch_save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise ProvenanceError(f"git {' '.join(args)}", out.stderr.strip() or f"exit code {out.returncode}")
    return out.stdout


def code_state(allow_dirty: bool, operation: str) -> dict:
    """Git commit and cleanliness. A dirty tree or an unborn branch is refused unless allow_dirty."""
    try:
        commit = _git("rev-parse", "HEAD").strip()
    except ProvenanceError as exc:
        if not allow_dirty:
            raise ProvenanceError(operation, f"cannot determine git commit ({exc.detail}); commit the code "
                                             f"or pass --allow-dirty") from exc
        commit = None
    changed = [line[3:] for line in _git("status", "--porcelain", "--untracked-files=normal").splitlines()
               if not line[3:].startswith("runs/")]
    if changed and not allow_dirty:
        raise DirtyTreeError(operation, changed)
    diff = _git("diff", "HEAD") if commit is not None and changed else ""
    return {"git_commit": commit, "git_dirty": bool(changed), "git_changed_paths": changed,
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "git_diff": diff if diff else None}


def runtime_info(device: torch.device | None) -> dict:
    from importlib.metadata import version

    info = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "scipy": version("scipy"), "optuna": version("optuna"), "pandas": version("pandas"),
            "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda,
            "host": socket.gethostname(), "platform": platform.platform(), "pid": os.getpid()}
    if device is not None and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info |= {"device": str(device), "gpu_name": props.name, "gpu_total_mb": props.total_memory // 2 ** 20,
                 "gpu_arch": getattr(props, "gcnArchName", None)}
    else:
        info |= {"device": str(device) if device is not None else None}
    return info


class StudyPaths:
    def __init__(self, root: Path):
        self.root = root

    @property
    def study_json(self) -> Path:
        return self.root / "study.json"

    @property
    def ledger(self) -> Path:
        return self.root / "ledger.sqlite"

    @property
    def ranges(self) -> Path:
        return self.root / "ranges.json"

    @property
    def pause_flag(self) -> Path:
        return self.root / "PAUSE"

    def job(self, job_id: str) -> "JobPaths":
        return JobPaths(self.root / "jobs" / job_id)


class JobPaths:
    def __init__(self, root: Path):
        self.root = root
        self.job_json = root / "job.json"
        self.events = root / "events.jsonl"
        self.metrics = root / "metrics"
        self.preds = root / "preds"
        self.tuning = root / "tuning"
        self.weights_init = root / "weights_init.pt"
        self.weights_final = root / "weights_final.pt"
        self.ckpt = root / "ckpt.pt"
        self.done = root / "DONE.json"

    def outputs(self) -> list[Path]:
        files = [p for p in self.root.rglob("*") if p.is_file()]
        return sorted(p for p in files if p.name not in ("DONE.json", "ckpt.pt", "events.jsonl")
                      and not p.name.endswith(".tmp"))
