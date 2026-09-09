"""
Figure 4 — day-to-day reproducibility of which PC pairs carry the map, for
arrays with enough oriented channels to be evaluated on multiple days.

Top row: pooled summary (L2 distance / top-3 Jaccard overlap of the 21-pair
error profile between days), across all qualifying arrays.
Bottom row: concrete per-array illustration — the per-pair percentile
profile on each recording day, for two representative example arrays, so
you can see directly which PC pair dips below the 5% line on each day and
how much that pair moves around.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lib
from lib import (MONKEYS, MONKEY_COLORS, TABLE_DIR, SIG_THRESH, MAX_DIM,
                  save_fig, set_pub_style, per_pair_percentile, all_pairs,
                  pair_label)

set_pub_style()

day_stab = pd.read_csv(os.path.join(TABLE_DIR, "day_stability.csv"))
with open(os.path.join(TABLE_DIR, "qualifying_stability_arrays.json")) as f:
    qualifying = json.load(f)

n_arrays_by_monkey = {m: sum(1 for (mm, a, d) in qualifying if mm == m) for m in MONKEYS}
n_pairs_by_monkey = day_stab.groupby("monkey").size().to_dict()

# example arrays for the bottom row: both fully estimable, 3 recording
# days each, so the shift in winning pair is shown against a clean backdrop
# (this is not cherry-picking a *pattern* -- these are the two clearest
# estimable arrays already highlighted in Figures 1-3; only the choice of
# WHICH estimable arrays to draw is ours, the profiles themselves are not)
EXAMPLES = [("L", 16), ("L", 13)]
STAB_COND = "all"
NORM = "L2"

compact = lib.load_compact()
REAL_DF, CTRL_VEC = compact["real"], compact["ctrl"]
cellsdf = pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv"))
MAIN_BIN = int(cellsdf.bin_ms.iloc[0])
stab_pairs = all_pairs(MAX_DIM)
stab_labels = [pair_label(*p) for p in stab_pairs]

fig = plt.figure(figsize=(7.2, 6.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.70, wspace=0.4,
                       top=0.82, bottom=0.10, left=0.09, right=0.97)

# # ---------- top row: pooled summary (A, B) ----------
# axA = fig.add_subplot(gs[0, 0])
# axB = fig.add_subplot(gs[0, 1])
# for ax, col, name, better in [
#         (axA, "l2", "L2 distance of per-pair\nerror profiles (deg)", "lower = more similar"),
#         (axB, "jaccard", "top-3 winning-pair\noverlap (Jaccard)", "higher = more similar")]:
#     for k, monkey in enumerate(MONKEYS):
#         vals = day_stab[day_stab.monkey == monkey][col].dropna().values
#         if not len(vals):
#             continue
#         xj = np.full(len(vals), k) + np.random.default_rng(0).uniform(-0.13, 0.13, len(vals))
#         ax.scatter(xj, vals, color=MONKEY_COLORS.get(monkey, "gray"), s=22,
#                    alpha=0.85, edgecolor="k", linewidth=0.25)
#         ax.hlines(np.median(vals), k - 0.22, k + 0.22, color="black", lw=1.6, zorder=5)
#     ax.set_xticks(range(len(MONKEYS)))
#     ax.set_xticklabels([f"{m}\n{n_arrays_by_monkey[m]} arr / {n_pairs_by_monkey.get(m, 0)} "
#                          f"pair{'s' if n_pairs_by_monkey.get(m, 0) != 1 else ''}"
#                          for m in MONKEYS], fontsize=6.5)
#     ax.set_ylabel(name, fontsize=7.5)
#     ax.set_title(f"pooled across arrays\n({better})", fontsize=7.5)
#     ax.grid(alpha=0.25, axis="y")

# ---------- left column: per-array per-pair percentile profile (A, B) ----------
x = np.arange(len(stab_pairs))
for col, (monkey, arr) in enumerate(EXAMPLES):
    ax = fig.add_subplot(gs[1, col])
    days = lib.DATES[monkey]["RS"]
    for dt in days:
        pp = per_pair_percentile(REAL_DF, CTRL_VEC, monkey, dt, arr, STAB_COND,
                                  MAIN_BIN, MAX_DIM, NORM)
        pct = [pp[p][1] for p in stab_pairs]
        best_k = int(np.nanargmin(pct))
        line, = ax.plot(x, pct, marker="o", ms=3, lw=1.3, label=str(dt), alpha=0.9)
        ax.scatter([best_k], [pct[best_k]], marker="*", s=110,
                   color=line.get_color(), edgecolor="k", linewidth=0.5, zorder=6)
    ax.axhline(SIG_THRESH, color="grey", ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(stab_labels, rotation=90, fontsize=5.5)
    ax.set_ylabel("per-pair percentile (%)", fontsize=7)
    ax.set_title(f"monkey {monkey}, array {arr}\n"
                 f"($\\star$ = best pair that day)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, title="day", title_fontsize=6, loc="upper right")
    ax.grid(alpha=0.2)

fig.suptitle("Day-to-day reproducibility of the map-carrying PC pairs\n"
             "(arrays with $\\geq$25 oriented channels, condition = all, bin = "
             f"{MAIN_BIN} ms, {NORM})",
             fontsize=9.5, y=0.995)

for ax, lab in zip([axA, axB], "AB"):
    ax.text(-0.22, 1.30, lab, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")
for col, lab in zip(range(2), "CD"):
    ax = fig.axes[2 + col]
    ax.text(-0.14, 1.20, lab, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")

save_fig(fig, "fig4_day_stability")
plt.close(fig)
