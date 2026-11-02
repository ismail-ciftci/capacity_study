"""Fixed visual outputs (guide Ch. 12): map lattices, region graphs, EDF panels, tuning lattices.

Static figures (PNG + PDF) on a light surface. Colors: diverging blue <-> red with a gray midpoint at 0 for test R2,
gap and T1 (one scale per quantity across all its figures); status uses the fixed status palette plus a letter in
every square so identity never rests on color alone; sequential blue for tuning values; untunable recipes are gray
with hatching, outside every scale.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_study.analysis.status import OVERFIT, STATUS_LABELS, UNSATURATED, UNTUNABLE, VALID
from capacity_study.config import StudyConfig

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIVERGING = ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a7a6", "#e34948", "#a32626"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS_COLORS = {VALID: "#0ca30c", UNSATURATED: "#fab219", OVERFIT: "#d03b3b", UNTUNABLE: MUTED}
STATUS_LETTER = {VALID: "V", UNSATURATED: "U", OVERFIT: "O", UNTUNABLE: "X"}
QUANTITY_TITLE = {"test_r2": "Test R² (seed mean)", "gap": "Generalization gap (seed mean)",
                  "t1": "T1 log ratio, test vs. train", "status": "Recipe status"}
QUANTITY_COLUMN = {"test_r2": "test_r2_mean", "gap": "gap_mean", "t1": "t1_logratio"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": "#c3c2b7", "font.family": ["Segoe UI", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 10})
    return plt


def _save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        names.append(f"{name}.{ext}")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return names


def symmetric_limit(values: pd.Series) -> float:
    v = values.replace([np.inf, -np.inf], np.nan).dropna().abs()
    lim = float(np.quantile(v, 0.98)) if len(v) else 1.0
    return lim if lim > 0 else 1.0


def _diverging(lim: float):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    return LinearSegmentedColormap.from_list("div", DIVERGING), Normalize(-lim, lim, clip=True)


def lattice(cfg: StudyConfig, table: pd.DataFrame, value_col: str | None, region_col: str | None,
            regularization: str, title: str, out: Path, name: str, *, cmap=None, norm=None, status=False,
            edge_cols: tuple[str, ...] = ("edge_hit_lr", "edge_hit_wd"), colorbar_label: str = "") -> list[str]:
    plt = _plt()
    g = cfg.grid
    nd, nw = len(g.depths), len(g.widths)
    fig, axes = plt.subplots(len(g.tiers), len(g.histories), figsize=(2.3 * len(g.histories) + 1.2,
                                                                      2.1 * len(g.tiers)), squeeze=False)
    sub = table[table["regularization"] == regularization].set_index(["tier", "history", "depth", "width"])
    for i, tier in enumerate(g.tiers):
        for j, hist in enumerate(g.histories):
            ax = axes[i][j]
            ax.set_xlim(-0.5, nw - 0.5)
            ax.set_ylim(nd - 0.5, -0.5)
            labels = np.zeros((nd, nw), dtype=int)
            for di, d in enumerate(g.depths):
                for wi, w in enumerate(g.widths):
                    row = sub.loc[(tier, hist, d, w)]
                    untunable = bool(row["untunable"])
                    if status:
                        color = STATUS_COLORS[row["status"]]
                    elif untunable:
                        color = MUTED
                    else:
                        color = cmap(norm(row[value_col]))
                    ax.add_patch(_rect(di, wi, color, hatch="///" if untunable else None))
                    text = []
                    if region_col is not None:
                        labels[di, wi] = int(row[region_col])
                        text.append(str(labels[di, wi]))
                    if status:
                        text.insert(0, STATUS_LETTER[row["status"]])
                    if text:
                        ax.text(wi, di, " ".join(text), ha="center", va="center", fontsize=6.5,
                                color=INK if status and row["status"] == UNSATURATED else _ink_on(color))
                    if any(bool(row[c]) for c in edge_cols if c in row.index):
                        ax.plot(wi + 0.36, di - 0.36, marker="^", markersize=3.5, linestyle="none",
                                color=INK if status and row["status"] == UNSATURATED else _ink_on(color))
            if region_col is not None:
                _borders(ax, labels)
            ax.set_xticks(range(nw), [str(w) for w in g.widths] if i == len(g.tiers) - 1 else [])
            ax.set_yticks(range(nd), [str(d) for d in g.depths] if j == 0 else [])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"history {hist}y", fontsize=9, color=INK_2)
            if j == 0:
                ax.set_ylabel(f"tier {tier}\ndepth", fontsize=8)
            if i == len(g.tiers) - 1:
                ax.set_xlabel("width", fontsize=8)
    fig.suptitle(f"{title} — regularization: {regularization}", x=0.02, ha="left", fontsize=11)
    if status:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=STATUS_COLORS[s], label=f"{STATUS_LETTER[s]} = {s}") for s in STATUS_LABELS]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    elif cmap is not None:
        from matplotlib.cm import ScalarMappable
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.5, pad=0.02)
        cb.set_label(colorbar_label, color=INK_2)
        cb.outline.set_visible(False)
    fig.text(0.02, -0.075 if status else -0.01,
             ("Numbers: region ID. " if region_col is not None else "")
             + "▲ edge hit in tuning. Gray hatched: untunable (outside the scale).",
             fontsize=7, color=MUTED)
    return _save(fig, out, name)


def _rect(di: int, wi: int, color, hatch=None):
    from matplotlib.patches import Rectangle
    return Rectangle((wi - 0.5, di - 0.5), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=1.0, hatch=hatch)


def _ink_on(color) -> str:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


def _borders(ax, labels: np.ndarray) -> None:
    nd, nw = labels.shape
    for di in range(nd):
        for wi in range(nw):
            if wi + 1 < nw and labels[di, wi] != labels[di, wi + 1]:
                ax.plot([wi + 0.5, wi + 0.5], [di - 0.5, di + 0.5], color=INK, linewidth=1.6)
            if di + 1 < nd and labels[di, wi] != labels[di + 1, wi]:
                ax.plot([wi - 0.5, wi + 0.5], [di + 0.5, di + 0.5], color=INK, linewidth=1.6)


def region_graph(cfg: StudyConfig, table: pd.DataFrame, regions: pd.DataFrame, region_col: str,
                 adjacency: dict[int, list[int]], title: str, out: Path, name: str, *, cmap=None, norm=None,
                 status=False) -> list[str]:
    """Nodes = regions at the centroid of their members on a fixed grid embedding (x: tier/history/reg,
    y: depth/width); size ~ recipe count; edges ~ number of adjacent recipe pairs between regions."""
    plt = _plt()
    g = cfg.grid
    t = table.set_index("recipe_id")
    def coords(rid):
        r = t.loc[rid]
        x = (g.tiers.index(r["tier"]) * (len(g.histories) + 1) + g.histories.index(r["history"])) \
            + 0.3 * g.regularizations.index(r["regularization"])
        y = g.depths.index(r["depth"]) * (len(g.widths) + 1) + g.widths.index(r["width"])
        return x, y
    pos = {}
    for reg_id, members in t.groupby(region_col).groups.items():
        xy = np.array([coords(rid) for rid in members])
        pos[reg_id] = xy.mean(axis=0)
    edges: dict[tuple[int, int], int] = {}
    for rid, ns in adjacency.items():
        for n in ns:
            a, b = int(t.loc[rid, region_col]), int(t.loc[n, region_col])
            if rid < n and a != b:
                key = (min(a, b), max(a, b))
                edges[key] = edges.get(key, 0) + 1
    fig, ax = plt.subplots(figsize=(9, 6))
    max_e = max(edges.values()) if edges else 1
    for (a, b), c in sorted(edges.items()):
        ax.plot(*zip(pos[a], pos[b]), color="#c3c2b7", linewidth=0.5 + 3.0 * c / max_e, zorder=1)
    reg = regions.set_index("region_id")
    for reg_id, (x, y) in sorted(pos.items()):
        r = reg.loc[reg_id]
        if status:
            color = STATUS_COLORS[r["label"]]
        elif r["untunable"]:
            color = MUTED
        else:
            color = cmap(norm(r["mu"]))
        ax.scatter([x], [y], s=140 + 18 * r["size"], color=color, edgecolors=SURFACE, linewidths=2, zorder=2)
        ax.annotate(str(reg_id), (x, y), ha="center", va="center", fontsize=7, color=_ink_on(color), zorder=3)
    ax.set_xticks([g.tiers.index(tr) * (len(g.histories) + 1) + 1 for tr in g.tiers],
                  [f"tier {tr}" for tr in g.tiers])
    ax.set_yticks([g.depths.index(d) * (len(g.widths) + 1) + 1.5 for d in g.depths],
                  [f"depth {d}" for d in g.depths])
    x_max = (len(g.tiers) - 1) * (len(g.histories) + 1) + len(g.histories) - 1 + 0.3
    y_max = (len(g.depths) - 1) * (len(g.widths) + 1) + len(g.widths) - 1
    ax.set_xlim(-1.0, x_max + 1.0)
    ax.set_ylim(y_max + 1.0, -1.0)  # depth 1 at the top, as in the lattices
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title}: region graph ({len(pos)} regions)", loc="left")
    if not status and cmap is not None:
        from matplotlib.cm import ScalarMappable
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
        cb.outline.set_visible(False)
    else:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=STATUS_COLORS[s], label=s) for s in STATUS_LABELS], frameon=False,
                  loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.text(0.01, -0.02, "Node position: centroid of members (x: tier, history, regularization offset; "
                          "y: depth, width). Node size: recipe count. Edge width: adjacent recipe pairs.",
             fontsize=7, color=MUTED)
    return _save(fig, out, name)


def edf_panel(cfg: StudyConfig, table: pd.DataFrame, quantity: str, factor: str, out: Path, name: str,
              xlim: float) -> list[str]:
    plt = _plt()
    col = QUANTITY_COLUMN[quantity]
    real = table[~table["untunable"]]
    options = list(getattr(cfg.grid, {"depth": "depths", "width": "widths", "regularization": "regularizations",
                                      "tier": "tiers", "history": "histories"}[factor]))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k, opt in enumerate(options):
        vals = np.sort(real.loc[real[factor] == opt, col].dropna().to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=CATEGORICAL[k], linewidth=2, label=f"{factor} {opt}")
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, 1)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(QUANTITY_TITLE[quantity])
    ax.set_ylabel("fraction of recipes ≤ value")
    n_unt = int(table["untunable"].sum())
    ax.set_title(f"EDF by {factor}" + (f" ({n_unt} untunable excluded)" if n_unt else ""), loc="left")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out, name)


def tuning_lattices(cfg: StudyConfig, table: pd.DataFrame, out: Path) -> list[str]:
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    seq = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    files = []
    real = table[~table["untunable"]]
    lr_norm = LogNorm(real["tuned_lr"].min(), real["tuned_lr"].max()) if len(real) else None
    wd = real.loc[real["regularization"] == "wd", "tuned_wd"]
    wd_norm = LogNorm(wd.min(), wd.max()) if len(wd) else None
    for reg in cfg.grid.regularizations:
        files += lattice(cfg, table, "tuned_lr", None, reg, "Tuned learning rate", out, f"tuning_lr_{reg}",
                         cmap=seq, norm=lr_norm, colorbar_label="learning rate (log)")
        files += lattice(cfg, table, "infeasible_fraction", None, reg, "Infeasible trial fraction", out,
                         f"tuning_infeasible_{reg}", cmap=seq, norm=Normalize(0, 1), colorbar_label="fraction")
    if "wd" in cfg.grid.regularizations:
        files += lattice(cfg, table, "tuned_wd", None, "wd", "Tuned weight decay λ", out, "tuning_wd_wd",
                         cmap=seq, norm=wd_norm, colorbar_label="λ (log)")
    return files


def all_figures(cfg: StudyConfig, table: pd.DataFrame, region_tables: dict[str, pd.DataFrame],
                adjacency: dict[int, list[int]], out: Path) -> list[str]:
    files = []
    real = table[~table["untunable"]]
    for q in ("test_r2", "gap", "t1"):
        cmap, norm = _diverging(symmetric_limit(real[QUANTITY_COLUMN[q]]))
        for reg in cfg.grid.regularizations:
            files += lattice(cfg, table, QUANTITY_COLUMN[q], f"region_{q}", reg, QUANTITY_TITLE[q], out,
                             f"map_{q}_{reg}", cmap=cmap, norm=norm, colorbar_label=QUANTITY_TITLE[q])
        files += region_graph(cfg, table, region_tables[q], f"region_{q}", adjacency, QUANTITY_TITLE[q], out,
                              f"regions_{q}", cmap=cmap, norm=norm)
        lim = symmetric_limit(real[QUANTITY_COLUMN[q]])
        for factor in ("depth", "width", "regularization", "tier", "history"):
            files += edf_panel(cfg, table, q, factor, out, f"edf_{q}_{factor}", lim * 1.05)
    for reg in cfg.grid.regularizations:
        files += lattice(cfg, table, None, "region_status", reg, QUANTITY_TITLE["status"], out,
                         f"map_status_{reg}", status=True)
    files += region_graph(cfg, table, region_tables["status"], "region_status", adjacency,
                          QUANTITY_TITLE["status"], out, "regions_status", status=True)
    files += tuning_lattices(cfg, table, out)
    return files
