"""
paper/panels.py
===============
Reusable panel-drawing functions shared by the main and supplementary
figures (trial-level percentile dot plot, channel-level RF/SNR panels,
OP-coverage / control-comparison panels). Each function draws into axes or
a subplot spec handed in by the figure script; data come from paper/tables/.
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import lib
from lib import (MONKEYS, MONKEY_COLORS, TABLE_DIR, SIG_THRESH, get_layout, load_op)

# validated categorical slots (dataviz reference palette), fixed order
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
COND_COLORS = {"all": CAT[0], "EC": CAT[1], "EO": CAT[2]}
FEAT_COLORS = {"rf_size": CAT[0], "snr": CAT[1]}
FEAT_LABELS = {"rf_size": "RF size (deg)", "snr": "SNR"}
FLOOR = 100.0 / lib.P["control"]["n_control"]  # percentile resolution


def _cells():
    cellsdf = pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv"))
    return cellsdf[cellsdf.norm == "L2"].copy()


def _channels():
    chan = pd.read_csv(os.path.join(TABLE_DIR, "channel_errors_L.csv"))
    with open(os.path.join(TABLE_DIR, "channel_stats_L.json")) as f:
        stats = json.load(f)
    return chan, stats


def _array_colors(chan):
    return dict(zip(sorted(chan.array.unique()), CAT))


# ─────────────────────────────────────────────────────────────────
# Trial-level min-over-pairs percentile, one band per included array
# ─────────────────────────────────────────────────────────────────

def draw_trial_dotplot(ax, marker_size=26, fontsize=7):
    sub = _cells()
    sub["pct_plot"] = sub.percentile.clip(lower=FLOOR)
    conds = ["all", "EC", "EO"]
    offset = {"all": -0.25, "EC": 0.0, "EO": 0.25}
    day_markers = ["o", "s", "^"]

    rows = []  # monkey blocks, arrays sorted by estimability within a monkey
    for m in MONKEYS:
        s = sub[sub.monkey == m]
        order = (s.groupby("array").percentile
                 .agg(frac=lambda p: np.mean(p < SIG_THRESH), med="median")
                 .sort_values(["frac", "med"], ascending=[False, True]).index)
        rows += [(m, int(a)) for a in order]

    ax.axvline(SIG_THRESH, color="crimson", ls="--", lw=1.0, zorder=1)
    ax.text(SIG_THRESH * 0.92, -0.78, "beats control", ha="right", va="center",
            fontsize=fontsize - 0.5, color="0.35")
    for y, (m, a) in enumerate(rows):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="0.94", zorder=0, lw=0)
        ax.axhline(y + 0.5, color="0.75", lw=0.5, zorder=1)
        s = sub[(sub.monkey == m) & (sub.array == a)]
        days = lib.DATES[m]["RS"]
        for _, r in s.iterrows():
            d_idx = days.index(str(r.date))
            # trials at the resolution floor coincide -> dodge them sideways by day
            x = r.pct_plot / (1.5 ** d_idx) if r.percentile <= FLOOR else r.pct_plot
            ax.scatter(x, y + offset[r.condition], s=marker_size, marker=day_markers[d_idx],
                       color=COND_COLORS[r.condition], edgecolor="0.15", linewidth=0.4,
                       zorder=3)
        n_sig, n = int((s.percentile < SIG_THRESH).sum()), len(s)
        ax.text(1.015, y, f"{n_sig}/{n}", transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=fontsize,
                fontweight="bold" if n_sig >= n / 2 else "normal")
    for k in range(1, len(rows)):
        if rows[k][0] != rows[k - 1][0]:
            ax.axhline(k - 0.5, color="0.3", lw=1.2, zorder=2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{m}{a}" for m, a in rows], fontsize=fontsize)
    for lab, (m, _) in zip(ax.get_yticklabels(), rows):
        lab.set_color(MONKEY_COLORS[m]); lab.set_fontweight("bold")
    ax.set_ylim(len(rows) - 0.5, -1.1)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR / 2.8, 100)
    ax.set_xticks([FLOOR, 0.1, 1, SIG_THRESH, 10, 100])
    ax.set_xticklabels([f"$\\leq${FLOOR:g}", "0.1", "1", f"{SIG_THRESH:g}", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("min-over-pairs percentile (%)  — lower = better than control")
    ax.text(1.015, -0.85, "# sig.\ntrials", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=fontsize - 0.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [Line2D([], [], marker="o", ls="", markersize=0.8 * fontsize, color=COND_COLORS[c],
                      markeredgecolor="0.15", markeredgewidth=0.4, label=c) for c in conds]
    handles += [Line2D([], [], marker=day_markers[i], ls="", markersize=0.8 * fontsize, color="0.6",
                       markeredgecolor="0.15", markeredgewidth=0.4, label=f"day {i + 1}")
                for i in range(max(len(lib.DATES[m]["RS"]) for m in MONKEYS))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6,
              fontsize=fontsize, frameon=False, handletextpad=0.2, columnspacing=1.0,
              title="condition (colour, sub-row) and recording day (shape)",
              title_fontsize=fontsize)
    return rows


# ─────────────────────────────────────────────────────────────────
# Channel-level panels (monkey L, included arrays)
# ─────────────────────────────────────────────────────────────────

MAP_SPECS = {"error": ("reconstruction error (deg)", (0, 90)),
             "rf_size": ("RF size (deg)", None),
             "snr": ("SNR", None)}


def draw_channel_maps(fig, spec, ex_array=14, fields=("error", "rf_size", "snr"),
                      wspace=0.55, title_fs=7.5, vertical=False, hspace=0.35):
    """Chosen channel properties of one array on the 8x8 grid; returns the axes."""
    chan, _ = _channels()
    ex = chan[chan.array == ex_array].set_index("channel")
    layout = get_layout("L", ex_array)
    specs = [(f, *MAP_SPECS[f]) for f in fields]
    shape = (len(specs), 1) if vertical else (1, len(specs))
    inner = gridspec.GridSpecFromSubplotSpec(*shape, subplot_spec=spec, wspace=wspace,
                                             hspace=hspace)
    axes = []
    for k, (col, lab, lim) in enumerate(specs):
        ax = fig.add_subplot(inner[k])
        vals = np.full(layout.shape, np.nan)
        for r in range(layout.shape[0]):
            for c in range(layout.shape[1]):
                if layout[r, c] in ex.index:
                    vals[r, c] = ex.loc[layout[r, c], col]
        vmin, vmax = lim if lim else (np.nanpercentile(vals, 2), np.nanpercentile(vals, 98))
        cm = plt.get_cmap("Blues").copy(); cm.set_bad("0.88")
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=cm, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        for i in range(layout.shape[1] + 1):
            ax.axvline(i - 0.5, color="white", lw=0.5)
            ax.axhline(i - 0.5, color="white", lw=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=title_fs - 2)
        ax.set_title(lab, fontsize=title_fs)
        axes.append(ax)
    lab_ax = axes[0] if not vertical else None
    if lab_ax is not None:
        lab_ax.text(-0.06, 0.5, f"array L{ex_array}", transform=lab_ax.transAxes,
                    rotation=90, ha="right", va="center", fontsize=title_fs - 0.5)
    else:
        top, bot = axes[0].get_position(), axes[-1].get_position()
        fig.text(top.x0 - 0.012, (top.y1 + bot.y0) / 2, f"array L{ex_array}", rotation=90,
                 ha="right", va="center", fontsize=title_fs - 0.5)
    return axes


def draw_within_array_scatter(ax, feat, legend=False, fontsize=7, marker_size=10):
    """Channel error vs `feat`, both relative to the array mean, pooled over arrays."""
    chan, stats = _channels()
    colors = _array_colors(chan)
    lab = FEAT_LABELS[feat]
    d = chan.dropna(subset=[feat]).copy()
    d["x"] = d[feat] - d.groupby("array")[feat].transform("mean")
    d["y"] = d.error - d.groupby("array").error.transform("mean")
    for arr, c in colors.items():
        s = d[d.array == arr]
        ax.scatter(s.x, s.y, s=marker_size, color=c, edgecolor="0.2", linewidth=0.25, alpha=0.85,
                   label=f"L{arr}", zorder=3)
    q = pd.qcut(d.x, 5)
    med = d.groupby(q, observed=True).agg(x=("x", "median"), y=("y", "median"))
    ax.plot(med.x, med.y, color="black", lw=1.5, marker="o", ms=0.5 * fontsize, zorder=5,
            label="quintile median")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    st = stats[feat]
    ax.set_title(f"{lab}: within-array $\\rho$ = {st['rho_within']:+.2f}\n"
                 f"p = {st['p']:.2g}, n = {st['n']} channels", fontsize=fontsize + 0.5)
    ax.set_xlabel(f"{lab}, relative to array mean")
    ax.set_ylabel("reconstruction error (deg),\nrelative to array mean")
    ax.grid(alpha=0.2)
    if legend:  # between title and axes, clear of the data
        ax.legend(fontsize=fontsize - 1, frameon=False, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2,
                  columnspacing=0.7, borderaxespad=0.2, markerscale=1.2)
        ax.set_title(ax.get_title(), fontsize=fontsize + 0.5, pad=4.3 * fontsize)


def draw_per_array_rho(ax, fontsize=7):
    chan, stats = _channels()
    arrays = sorted(chan.array.unique())
    x = np.arange(len(arrays)); w = 0.38
    for j, feat in enumerate(["rf_size", "snr"]):
        rhos = [stats[feat]["per_array_rho"][str(a)] for a in arrays]
        ax.bar(x + (j - 0.5) * w, rhos, width=w * 0.92, color=FEAT_COLORS[feat],
               edgecolor="0.2", linewidth=0.3, label=FEAT_LABELS[feat])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{a}\n({int((chan.array == a).sum())} ch)" for a in arrays],
                       fontsize=fontsize - 0.5)
    ax.set_ylabel("Spearman $\\rho$ between channel\nreconstruction error and RF size / SNR")
    ax.set_title("same correlation, computed within each array\n"
                 "(> 0: larger value = worse reconstruction)", fontsize=fontsize + 0.5)
    ax.set_ylim(-0.32, 0.52)
    ax.legend(fontsize=fontsize - 0.5, frameon=False, loc="upper center", ncol=2,
              handlelength=1.0, columnspacing=0.8)
    ax.grid(alpha=0.2, axis="y")


# ─────────────────────────────────────────────────────────────────
# OP coverage and choice of control (all 29 arrays)
# ─────────────────────────────────────────────────────────────────

def _coverage():
    cov = pd.read_csv(os.path.join(TABLE_DIR, "op_coverage_per_array.csv"))
    cells = pd.read_csv(os.path.join(TABLE_DIR, "percentile_uniform_vs_perm.csv"))
    return cov, cells.merge(cov[["monkey", "array", "op_concentration"]])


def draw_op_roses(fig, spec, examples=(("L", 14), ("L", 10))):
    cov, _ = _coverage()
    inner = gridspec.GridSpecFromSubplotSpec(len(examples), 1, subplot_spec=spec, hspace=0.55)
    axes = []
    for k, (m, a) in enumerate(examples):
        ax = fig.add_subplot(inner[k], projection="polar")
        op = load_op(m, a); op = op[op >= 0]
        edges = np.deg2rad(np.arange(0, 181, 15))
        cnt, _ = np.histogram(np.deg2rad(op), edges)
        ax.bar(edges[:-1], cnt, width=np.diff(edges), align="edge", color=CAT[0],
               edgecolor="white", linewidth=0.6)
        ax.set_thetamin(0); ax.set_thetamax(180)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
        ax.set_xticklabels(["0", "45", "90", "135", "180"], fontsize=5.5)
        ax.set_yticks([]); ax.grid(alpha=0.3)
        r = cov.set_index(["monkey", "array"]).loc[(m, a), "op_concentration"]
        ax.set_title(f"{m}{a}: concentration = {r:.2f}", fontsize=7, pad=2)
        axes.append(ax)
    return axes


def draw_coverage_scatter(ax):
    cov, _ = _coverage()
    for m in MONKEYS:
        s_ = cov[cov.monkey == m]
        ax.scatter(s_.op_concentration, 100 * s_.frac_sig_perm,
                   s=[34 if i else 18 for i in s_.included], color=MONKEY_COLORS[m],
                   edgecolor=["black" if i else "0.4" for i in s_.included],
                   linewidth=[1.0 if i else 0.3 for i in s_.included], alpha=0.9,
                   label=m, zorder=3)
    rho_p, p_p = spearmanr(cov.op_concentration, cov.frac_sig_perm)
    rho_u, p_u = spearmanr(cov.op_concentration, cov.frac_sig_uniform)
    ax.axhline(50, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("OP concentration\n(0 = orientations evenly covered)")
    ax.set_ylabel("% trials significant\n(permutation control)")
    ax.set_title(f"all 29 arrays: $\\rho$ = {rho_p:+.2f}, p = {p_p:.2g}\n"
                 f"(uniform control: $\\rho$ = {rho_u:+.2f}, p = {p_u:.2g})", fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right",
              title="monkey\n(black edge =\nincluded)", title_fontsize=6)
    ax.grid(alpha=0.2)


def draw_control_comparison(fig, ax):
    _, cells = _coverage()
    xs, ys = cells.pct_uniform.clip(lower=FLOOR), cells.pct_perm.clip(lower=FLOOR)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "conc", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
    sc = ax.scatter(xs, ys, c=cells.op_concentration, cmap=cmap, s=12, edgecolor="0.2",
                    linewidth=0.2, zorder=3)
    ax.plot([FLOOR, 100], [FLOOR, 100], color="black", lw=0.7, ls="--")
    ax.axvline(SIG_THRESH, color="crimson", lw=0.7, ls=":")
    ax.axhline(SIG_THRESH, color="crimson", lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(FLOOR / 1.5, 120); ax.set_ylim(FLOOR / 1.5, 120)
    ax.set_aspect("equal")
    ticks, labels = [0.1, 1, SIG_THRESH, 10, 100], ["0.1", "1", "5", "10", "100"]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.minorticks_off()
    same = int(((cells.pct_uniform < SIG_THRESH) == (cells.pct_perm < SIG_THRESH)).sum())
    ax.set_title(f"same verdict in {same}/{len(cells)} trials", fontsize=7.5)
    ax.set_xlabel("percentile, uniform control (%)")
    ax.set_ylabel("percentile, permutation control (%)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("OP concentration", fontsize=6.5); cb.ax.tick_params(labelsize=6)
