"""`analyze`: raw job outputs -> quantities, status, regions, characteristics, figures, provenance.

Output: <study>/analysis/<commit12>/ with tables/*.csv, per_seed.parquet, figures/*, analysis.json. Never writes
outside that directory and never modifies job outputs.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_study.analysis.characteristics import data_saturation, failure_rates, fanova, lr_vs_batch
from capacity_study.analysis.load import load_recipes
from capacity_study.analysis.quantities import check_sentinels, recipe_tables
from capacity_study.analysis.regions import Member, label_components, regionalize
from capacity_study.analysis.status import classify, flatness_threshold, margins
from capacity_study.errors import OverwriteRefusedError, StudyStateError
from capacity_study.orchestrate import load_study
from capacity_study.recipes import adjacency
from capacity_study.store import StudyPaths, atomic_write_json, code_state, sha256_path, utc_now

FACTORS = ("depth", "width", "regularization", "tier", "history")


def _spans(table: pd.DataFrame, ids: list[int]) -> dict:
    sub = table.set_index("recipe_id").loc[ids]
    return {f: sorted(sub[f].unique().tolist()) for f in FACTORS}


def _region_table(table: pd.DataFrame, reg) -> pd.DataFrame:
    rows = [{"region_id": rid, "size": len(r.members), "mu": r.mu, "sigma": r.sigma,
             "heterogeneity": r.heterogeneity, "untunable": r.untunable, "members": json.dumps(r.ids),
             "spans": json.dumps(_spans(table, r.ids))} for rid, r in reg.regions.items()]
    return pd.DataFrame(rows)


def analyze(root: Path, allow_dirty: bool, overwrite: bool, make_figures: bool = True) -> Path:
    cfg, config_hash, _ = load_study(root)
    paths = StudyPaths(root)
    code = code_state(allow_dirty, "analyze")
    out = root / "analysis" / ((code["git_commit"] or "uncommitted")[:12] + ("-dirty" if code["git_dirty"] else ""))
    if out.exists():
        if not overwrite:
            raise OverwriteRefusedError(str(out), "analyze")
        shutil.rmtree(out)
    tables_dir, fig_dir = out / "tables", out / "figures"
    tables_dir.mkdir(parents=True)

    raws, sources = load_recipes(cfg, paths)
    table, seeds = recipe_tables(cfg, raws)
    check_sentinels(cfg, table)
    warnings = []
    unstable = table.loc[table["n_nonfinite_seeds"] > 0, "recipe_id"].tolist()
    if unstable:
        warnings.append(f"{len(unstable)} recipes had final-stage seeds go non-finite; those seeds are excluded "
                        f"(guide does not cover this): {unstable}")
    lacking = table.loc[~table["untunable"] & table[["test_r2_sd", "gap_sd", "t1_logratio_se"]].isna().any(axis=1),
                        "recipe_id"].tolist()
    if lacking:
        raise StudyStateError(str(root), "analyze", f"recipes with too few finite seeds for seed statistics: "
                                                    f"{lacking}")

    m = margins(cfg.seeds.n_seeds, cfg.analysis.alpha, cfg.analysis.beta)
    flat_thr = flatness_threshold(table, m)
    table["status"] = classify(table, flat_thr, m)

    adj = adjacency(cfg)
    region_tables: dict[str, pd.DataFrame] = {}
    merge_logs = {}
    for q, value, spread, rule, delta in (("test_r2", "test_r2_mean", "test_r2_sd", "mean", m.delta_mu),
                                          ("gap", "gap_mean", "gap_sd", "mean", m.delta_mu),
                                          ("t1", "t1_logratio", "t1_logratio_se", "logratio", m.delta_sigma)):
        members = [Member(int(r.recipe_id), float(getattr(r, value)), float(getattr(r, spread)), int(r.n_seeds),
                          bool(r.untunable)) for r in table.itertuples()]
        reg = regionalize(members, adj, rule, delta, cfg.analysis.alpha)
        table[f"region_{q}"] = table["recipe_id"].map(reg.labels)
        region_tables[q] = _region_table(table, reg)
        merge_logs[q] = reg.merges
    status_labels = dict(zip(table["recipe_id"], table["status"]))
    comp = label_components(status_labels, adj)
    table["region_status"] = table["recipe_id"].map(comp)
    by_region: dict[int, list[int]] = {}
    for recipe_id, region_id in sorted(comp.items()):
        by_region.setdefault(region_id, []).append(recipe_id)
    region_tables["status"] = pd.DataFrame(
        [{"region_id": rid, "label": status_labels[ids[0]], "size": len(ids), "members": json.dumps(ids),
          "spans": json.dumps(_spans(table, ids)), "untunable": status_labels[ids[0]] == "untunable"}
         for rid, ids in sorted(by_region.items())])

    single, pairs = failure_rates(table)
    saturation = data_saturation(cfg, table, m.delta_mu)
    fanova_tables = {}
    if table["untunable"].any():
        warnings.append(f"functional ANOVA skipped: {int(table['untunable'].sum())} untunable recipes carry "
                        f"sentinel values that would dominate the variance")
    else:
        for col in ("test_r2_mean", "test_r2_sd", "gap_mean", "gap_sd", "t1_logratio"):
            fanova_tables[col] = fanova(cfg, table, col)
    edge_share = float((table["edge_hit_lr"] | table["edge_hit_wd"]).mean())

    written = {"recipes.csv": table, "failure_rates_option.csv": single, "failure_rates_pair.csv": pairs,
               "data_saturation.csv": saturation, "lr_vs_batch.csv": lr_vs_batch(table)}
    written |= {f"regions_{q}.csv": t for q, t in region_tables.items()}
    written |= {f"fanova_{c}.csv": t for c, t in fanova_tables.items()}
    for name, frame in written.items():
        frame.to_csv(tables_dir / name, index=False)
    seeds.to_parquet(tables_dir / "per_seed.parquet", index=False)
    atomic_write_json(tables_dir / "merge_history.json", merge_logs)

    figures = []
    if make_figures:
        from capacity_study.analysis.figures import all_figures
        figures = all_figures(cfg, table, region_tables, adj, fig_dir)

    inputs_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    outputs = {str(p.relative_to(out)).replace("\\", "/"): sha256_path(p) for p in sorted(out.rglob("*"))
               if p.is_file()}
    record = {"created_utc": utc_now(), "config_hash": config_hash,
              "code": {k: v for k, v in code.items() if k != "git_diff"}, "inputs_done_sha256": sources,
              "inputs_digest": inputs_digest,
              "margins": {"n": m.n, "alpha": m.alpha, "beta": m.beta, "delta_mu": m.delta_mu,
                          "delta_sigma": m.delta_sigma, "overfit_t1_ratio": float(np.exp(2 * m.delta_sigma))},
              "flatness_threshold": flat_thr,
              "counts": {"recipes": len(table), "status": table["status"].value_counts().to_dict(),
                         "regions": {q: len(t) for q, t in region_tables.items()},
                         "untunable": int(table["untunable"].sum()), "edge_hit_share": edge_share},
              "warnings": warnings, "figures": figures, "outputs_sha256": outputs}
    atomic_write_json(out / "analysis.json", record)
    return out
