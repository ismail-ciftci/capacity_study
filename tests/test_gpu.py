"""GPU checks (skipped without a GPU): deterministic resume on the device, and no host<->device traffic inside
the training steps."""
import pandas as pd
import pytest
import torch

from capacity_study import orchestrate
from capacity_study.recipes import pack_jobs
from capacity_study.store import StudyPaths
from capacity_study.train import read_metrics, run_job

from test_pipeline import _setup_study, _single_job_ctx

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")


def _on_gpu(ctx):
    ctx.device = torch.device("cuda:0")
    ctx.bank = ctx.bank.to(ctx.device)
    return ctx


def test_gpu_resume_bit_identical(tiny, tmp_path):
    torch.use_deterministic_algorithms(True)
    cfg, cfg_path = tiny
    root_a = _setup_study(cfg, cfg_path, tmp_path)
    spec = pack_jobs(cfg, "calibration")[-1]
    assert run_job(_on_gpu(_single_job_ctx(cfg, root_a, spec))) == "done"
    root_b = tmp_path / "runs" / "b"
    orchestrate.init_study(cfg_path, root_b, allow_dirty=True)
    assert run_job(_on_gpu(_single_job_ctx(cfg, root_b, spec, pause_at=3))) == "paused"
    assert run_job(_on_gpu(_single_job_ctx(cfg, root_b, spec))) == "done"
    a, b = StudyPaths(root_a).job(spec.job_id), StudyPaths(root_b).job(spec.job_id)
    pd.testing.assert_frame_equal(read_metrics(a, cfg.training.epochs), read_metrics(b, cfg.training.epochs))


def test_training_steps_have_no_host_transfers(tiny, tmp_path):
    """Profile a whole job; memcpy events may only come from setup, eval-free flushes and checkpoints, never
    from a per-step pattern: their count must not scale with the number of optimizer steps."""
    cfg, cfg_path = tiny
    root = _setup_study(cfg, cfg_path, tmp_path)
    spec = pack_jobs(cfg, "calibration")[-1]
    from torch.profiler import ProfilerActivity, profile
    counts = []
    for i, steps in enumerate((3, 9)):
        c = cfg.__class__(**{**vars(cfg), "training": cfg.training.__class__(
            **{**vars(cfg.training), "steps_per_epoch": steps})})
        ctx = _on_gpu(_single_job_ctx(c, root, spec))
        ctx.config_hash = f"{ctx.config_hash}-{i}"
        ctx.study = StudyPaths(tmp_path / "runs" / f"p{i}")
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            run_job(ctx)
        counts.append(sum(e.count for e in prof.key_averages()
                          if e.key in ("aten::_local_scalar_dense", "aten::item")))
    assert counts[0] == counts[1], counts  # syncs do not grow with steps per epoch
