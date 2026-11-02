import json
import math

import numpy as np
import pandas as pd
import pytest

from capacity_study import orchestrate
from capacity_study.analysis.characteristics import fanova
from capacity_study.analysis.quantities import jackknife_se, t1_logratio
from capacity_study.analysis.regions import Member, label_components, merge_statistic, Region, regionalize
from capacity_study.analysis.run import analyze
from capacity_study.analysis.status import OVERFIT, UNSATURATED, UNTUNABLE, VALID, classify, margins
from capacity_study.config import config_from_dict
from capacity_study.recipes import grid

from conftest import tiny_config_dict
from test_pipeline import _setup_study, _work


def test_margins_match_guide():
    m = margins(32, 0.05, 0.2)
    assert abs(m.delta_mu - 0.74) < 0.005          # guide Ch.8: ~0.74 sigma
    assert abs(m.delta_sigma - 0.53) < 0.005       # ~0.53
    assert abs(math.exp(2 * m.delta_sigma) - 2.9) < 0.1   # Ch.10: T1 ratio ~2.9


def test_fanova_recovers_additive_and_interaction_structure(tmp_path):
    cfg = config_from_dict(tiny_config_dict(tmp_path), "t")
    rows = []
    for r in grid(cfg):
        value = 2.0 * (r.depth == 2) + 1.0 * (r.regularization == "wd") * (r.tier == 2)
        rows.append({"depth": r.depth, "width": r.width, "regularization": r.regularization, "tier": r.tier,
                     "history": r.history, "v": value})
    out = fanova(cfg, pd.DataFrame(rows), "v").set_index("effect")
    assert abs(out["share"].sum() - 1.0) < 1e-12
    assert out.loc["width", "share"] == 0 and out.loc["history", "share"] == 0
    assert out.loc["regularization x tier", "share"] > 0
    assert out.loc[[e for e in out.index if out.loc[e, "order"] >= 3], "share"].sum() < 1e-12


def _chain(n):
    return {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}


def test_regionalize_finds_plateaus_and_respects_contiguity():
    values = [0.0] * 4 + [1.0] * 4 + [0.0] * 4          # third plateau equals the first but is not adjacent
    members = [Member(i, v, 0.05, 32, False) for i, v in enumerate(values)]
    reg = regionalize(members, _chain(12), "mean", margins(32, 0.05, 0.2).delta_mu, 0.05)
    assert [reg.labels[i] for i in range(12)] == [1] * 4 + [2] * 4 + [3] * 4
    assert len(reg.merges) == 9


def test_untunable_merge_rules():
    a, b = Region([Member(0, -1000, 0, 0, True)]), Region([Member(1, -1000, 0, 0, True)])
    c = Region([Member(2, 0.01, 0.02, 32, False)])
    assert merge_statistic("mean", a, b, 0.74, 0.05) == 0.0
    assert merge_statistic("mean", a, c, 0.74, 0.05) == math.inf


def test_label_components():
    labels = {0: "a", 1: "a", 2: "b", 3: "a"}
    assert label_components(labels, _chain(4)) == {0: 1, 1: 1, 2: 2, 3: 3}


def test_t1_logratio_and_jackknife():
    rng = np.random.default_rng(0)
    p_train = rng.normal(0, 0.01, (32, 200))
    p_test = rng.normal(0, 0.02, (32, 100))
    assert abs(t1_logratio(p_train, p_test) - math.log(2)) < 0.05      # sd ratio 2
    assert 0 < jackknife_se(p_train, p_test) < 0.1


def test_status_precedence():
    m = margins(32, 0.05, 0.2)
    table = pd.DataFrame([
        {"untunable": False, "v1a_abs_mean": 0.1, "v1b_abs_mean": 0.0, "t1_logratio": 2.0},
        {"untunable": False, "v1a_abs_mean": 0.0, "v1b_abs_mean": 0.0, "t1_logratio": 2.0},
        {"untunable": False, "v1a_abs_mean": 0.0, "v1b_abs_mean": 0.0, "t1_logratio": 0.1},
        {"untunable": True, "v1a_abs_mean": 1000, "v1b_abs_mean": 1000, "t1_logratio": 1000},
    ])
    assert classify(table, 0.01, m).tolist() == [UNSATURATED, OVERFIT, VALID, UNTUNABLE]


def test_analyze_end_to_end(tiny, tmp_path):
    cfg, cfg_path = tiny
    root = _setup_study(cfg, cfg_path, tmp_path)
    orchestrate.enqueue(root, "calibration")
    assert _work(root) == 0
    orchestrate.freeze(root)
    orchestrate.enqueue(root, "tuning")
    assert _work(root) == 0

    out = analyze(root, allow_dirty=True, overwrite=False)
    record = json.loads((out / "analysis.json").read_text())
    recipes = pd.read_csv(out / "tables" / "recipes.csv")
    assert len(recipes) == len(grid(cfg))
    assert set(recipes["status"]) <= {VALID, UNSATURATED, OVERFIT, UNTUNABLE}
    for q in ("test_r2", "gap", "t1", "status"):
        assert recipes[f"region_{q}"].notna().all()
    g = cfg.grid
    n_reg, n_factors = len(g.regularizations), 5
    pngs = [f for f in record["figures"] if f.endswith(".png")]
    assert len(pngs) == 4 * n_reg + 4 + 3 * n_factors + (2 * n_reg + 1)   # 8 + 4 + 15 + 5 for the full grid
    assert all((out / rel).exists() for rel in record["outputs_sha256"])
    with pytest.raises(Exception):
        analyze(root, allow_dirty=True, overwrite=False)      # never silently overwrites
