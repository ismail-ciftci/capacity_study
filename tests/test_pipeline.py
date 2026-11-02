"""End-to-end on CPU with a tiny grid and synthetic prices: calibration -> ranges -> tuning -> final,
plus pause/resume bit-identity and ledger semantics."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from capacity_study import orchestrate
from capacity_study.data.bank import prepare_bank
from capacity_study.ledger import Ledger
from capacity_study.recipes import grid, pack_jobs
from capacity_study.store import StudyPaths, code_state, runtime_info
from capacity_study.train import JobContext, read_metrics, run_job


def _setup_study(cfg, cfg_path: Path, tmp: Path) -> Path:
    prepare_bank(cfg)
    root = tmp / "runs" / "s"
    orchestrate.init_study(cfg_path, root, allow_dirty=True)
    return root


def _work(root: Path) -> int:
    return orchestrate.worker_main(root, "cpu", 1e9, allow_dirty=True, allow_code_change=False,
                                   max_consecutive_failures=1, idle_poll_seconds=0.01, heartbeat_seconds=0.0)


def test_end_to_end(tiny, tmp_path):
    cfg, cfg_path = tiny
    root = _setup_study(cfg, cfg_path, tmp_path)
    paths = StudyPaths(root)

    assert orchestrate.enqueue(root, "calibration") == 4
    assert _work(root) == 0
    orchestrate.freeze(root)
    ranges = json.loads(paths.ranges.read_text())["ranges"]
    assert ranges["lr"][0] < ranges["lr"][1]

    n_tuning = len(pack_jobs(cfg, "tuning"))
    assert orchestrate.enqueue(root, "tuning") == 2 * n_tuning
    assert _work(root) == 0

    ledger = Ledger(paths.ledger)
    states = {(p, s): c for p, s, c in ledger.status_counts()}
    ledger.close()
    assert states == {("calibration", "done"): 4, ("tuning", "done"): n_tuning, ("final", "done"): n_tuning}

    for spec in pack_jobs(cfg, "final"):
        jp = paths.job(spec.job_id)
        done = json.loads(jp.done.read_text())
        job = json.loads(jp.job_json.read_text())
        assert job["tuning_job"] == spec.job_id.replace("final-", "tuning-")
        if done["summary"].get("empty"):
            continue
        df = read_metrics(jp, cfg.training.epochs)
        assert len(df) == cfg.training.epochs * done["summary"]["members"]
        assert sorted(p.name for p in jp.preds.glob("*.pt")) == ["e0002.pt", "e0004.pt", "e0006.pt"]
        assert not jp.ckpt.exists()
        for rel, sha in done["outputs_sha256"].items():
            assert (jp.root / rel).exists(), rel


def _single_job_ctx(cfg, root, spec, pause_at=None):
    from capacity_study.orchestrate import _resolve, load_study
    _, config_hash, record = load_study(root)
    paths = StudyPaths(root)
    from capacity_study.data.bank import load_bank
    bank = load_bank(cfg, record["bank"]["key"], record["bank"]["sha256"])
    members, meta, ranges = _resolve(cfg, paths, spec)
    state = {"epochs": 0}

    def heartbeat():
        state["epochs"] += 1

    def should_pause():
        return pause_at is not None and state["epochs"] == pause_at

    return JobContext(cfg, config_hash, paths, spec, members, meta, bank, torch.device("cpu"),
                      code_state(True, "test"), runtime_info(None), False, should_pause, heartbeat, ranges)


def test_pause_resume_is_bit_identical(tiny, tmp_path):
    torch.use_deterministic_algorithms(True)
    cfg, cfg_path = tiny
    root_a = _setup_study(cfg, cfg_path, tmp_path)
    spec = pack_jobs(cfg, "calibration")[-1]
    assert run_job(_single_job_ctx(cfg, root_a, spec)) == "done"

    root_b = tmp_path / "runs" / "b"
    orchestrate.init_study(cfg_path, root_b, allow_dirty=True)
    assert run_job(_single_job_ctx(cfg, root_b, spec, pause_at=3)) == "paused"   # odd epoch: off-boundary flush
    assert StudyPaths(root_b).job(spec.job_id).ckpt.exists()
    assert run_job(_single_job_ctx(cfg, root_b, spec)) == "done"

    a, b = StudyPaths(root_a).job(spec.job_id), StudyPaths(root_b).job(spec.job_id)
    pd.testing.assert_frame_equal(read_metrics(a, cfg.training.epochs), read_metrics(b, cfg.training.epochs))
    wa = torch.load(a.weights_final, weights_only=True)
    wb = torch.load(b.weights_final, weights_only=True)
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    events = [json.loads(line)["event"] for line in b.events.read_text().splitlines()]
    assert events[:2] == ["start", "checkpoint"] and "pause" in events and "resume" in events


def test_trials_share_init_seeds_differ(tiny, tmp_path):
    cfg, cfg_path = tiny
    root = _setup_study(cfg, cfg_path, tmp_path)
    spec = pack_jobs(cfg, "calibration")[0]
    ctx = _single_job_ctx(cfg, root, spec)
    keys_by_recipe = {}
    for m in ctx.members:
        keys_by_recipe.setdefault(m.recipe_id, set()).add(m.stream_key)
    assert all(len(keys) == 1 for keys in keys_by_recipe.values())       # trials share init + order
    assert len({next(iter(k)) for k in keys_by_recipe.values()}) == 2    # recipes independent


_CLAIM_SCRIPT = """
import sys
from pathlib import Path
from capacity_study.ledger import Ledger
ledger = Ledger(Path(sys.argv[1]))
got = []
while True:
    job = ledger.claim(sys.argv[2], "dev", 1e12)
    if job is None:
        break
    got.append(job.spec.job_id)
print("\\n".join(got))
"""


def test_ledger_never_double_claims(tiny, tmp_path):
    cfg, cfg_path = tiny
    root = _setup_study(cfg, cfg_path, tmp_path)
    orchestrate.enqueue(root, "calibration")
    ledger = Ledger(StudyPaths(root).ledger)
    from capacity_study.recipes import JobSpec
    extra = [(JobSpec(f"x-{i}", "calibration", 1, 1, 1, 8, (0,)), float(i), 1.0, None) for i in range(60)]
    ledger.add_jobs(extra)
    ledger.close()
    procs = [subprocess.Popen([sys.executable, "-c", _CLAIM_SCRIPT, str(StudyPaths(root).ledger), f"w{i}"],
                              stdout=subprocess.PIPE, text=True) for i in range(8)]
    claimed = [line for p in procs for line in p.communicate()[0].split() if line]
    assert len(claimed) == 64 and len(set(claimed)) == 64


def test_ledger_reconcile_and_ownership(tiny, tmp_path):
    cfg, cfg_path = tiny
    root = _setup_study(cfg, cfg_path, tmp_path)
    paths = StudyPaths(root)
    orchestrate.enqueue(root, "calibration")
    ledger = Ledger(paths.ledger)
    job = ledger.claim("w1", "dev", 1e12)
    assert ledger.reconcile(paths, stale_seconds=3600)["reclaimed_stale"] == 0
    assert ledger.reconcile(paths, stale_seconds=-1)["reclaimed_stale"] == 1
    from capacity_study.errors import JobStateError
    with pytest.raises(JobStateError):
        ledger.complete(job.spec.job_id, "w1")       # lost ownership after reclaim
    # Partial output without DONE is never complete; DONE without a ledger update is reconciled.
    jp = paths.job(job.spec.job_id)
    jp.root.mkdir(parents=True)
    (jp.root / "job.json").write_text("{}")
    assert ledger.reconcile(paths, 3600)["marked_done"] == 0
    jp.done.write_text("{}")
    assert ledger.reconcile(paths, 3600)["marked_done"] == 1
    ledger.close()


def test_final_job_waits_for_tuning(tiny, tmp_path):
    cfg, cfg_path = tiny
    root = _setup_study(cfg, cfg_path, tmp_path)
    paths = StudyPaths(root)
    paths.ranges.write_text(json.dumps({"ranges": {"lr": [1e-4, 1e-2], "wd": [1e-4, 1e-1]}}))
    orchestrate.enqueue(root, "tuning")
    ledger = Ledger(paths.ledger)
    phases = []
    while (job := ledger.claim("w", "dev", 1e12)) is not None:
        phases.append(job.spec.phase)
    ledger.close()
    assert set(phases) == {"tuning"}


def test_grid_size_and_ids():
    from capacity_study.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "study.toml")
    recipes = grid(cfg)
    assert len(recipes) == 480 and [r.recipe_id for r in recipes] == list(range(480))
    assert len(pack_jobs(cfg, "tuning")) == 240 and len(pack_jobs(cfg, "calibration")) == 12
    from capacity_study.recipes import adjacency
    adj = adjacency(cfg)
    assert abs(sum(map(len, adj.values())) / 480 - 6.9) < 0.05     # guide Ch.12: ~6.9 neighbours on average
    assert all(i in adj[j] for i in adj for j in adj[i])
