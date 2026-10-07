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
EXAMPLES = [("L", 16), ("L", 14)]
STAB_COND = "all"
NORM = "L2"

compact = lib.load_compact()
REAL_DF, CTRL_VEC = compact["real"], compact["ctrl"]
cellsdf = pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv"))
MAIN_BIN = int(cellsdf.bin_ms.iloc[0])
stab_pairs = all_pairs(MAX_DIM)
stab_labels = [pair_label(*p) for p in stab_pairs]

pca_store = lib.load_pca_store()
eceo_df = pd.read_csv(os.path.join(TABLE_DIR, "eceo_comparison.csv"))
MAIN_BIN = int(pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv")).bin_ms.iloc[0])
NORM = "L2"

# cells significant in EC and/or EO -> tally which PC pair is significant
qual = eceo_df[(eceo_df.pct_EC < SIG_THRESH) | (eceo_df.pct_EO < SIG_THRESH)]
eceo_pairs = all_pairs(MAX_DIM)
labels = [pair_label(*p) for p in eceo_pairs]
tally = {"EC": np.zeros(len(eceo_pairs)), "EO": np.zeros(len(eceo_pairs))}
for _, row in qual.iterrows():
    for cond in ["EC", "EO"]:
        pp = per_pair_percentile(REAL_DF, CTRL_VEC, row["monkey"], row["date"], row["array"],
                                  cond, MAIN_BIN, MAX_DIM, NORM)
        for k, p in enumerate(eceo_pairs):
            _, pct, _ = pp[p]
            if not np.isnan(pct) and pct < SIG_THRESH:
                tally[cond][k] += 1


fig = plt.figure(figsize=(9, 6.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1., 1.,1.], height_ratios=[1., 1.], hspace=0.78,
                       wspace=0.32, top=0.91, bottom=0.08, left=0.08, right=0.97)

cols = ["#4D6566", "#FFCE14", "#83CC89"]

# ---------- left column: per-array per-pair percentile profile (A, B) ----------
x = np.arange(len(stab_pairs))
for col, (monkey, arr) in enumerate(EXAMPLES):
    ax = fig.add_subplot(gs[0, col])
    days = lib.DATES[monkey]["RS"]
    for dnr, dt in enumerate(days):
        pp = per_pair_percentile(REAL_DF, CTRL_VEC, monkey, dt, arr, STAB_COND,
                                  MAIN_BIN, MAX_DIM, NORM)
        pct = [pp[p][1] for p in stab_pairs]
        best_k = int(np.nanargmin(pct))
        line, = ax.plot(x, pct, marker="o", ms=3, lw=1.3, label=str(dt), alpha=0.9, color=cols[dnr])
        ax.scatter([best_k+0.3*dnr], [pct[best_k]+0.3*dnr], marker="*", s=110,
                   color=line.get_color(), edgecolor="k", linewidth=0.5, zorder=6)
    ax.axhline(SIG_THRESH, color="grey", ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(stab_labels, rotation=90, fontsize=5.5)
    ax.set_ylabel("percentile", fontsize=7)
    ax.set_title(f"monkey {monkey}, array {arr}\n"
                 f"($\\star$ = best pair that day)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, title="day", title_fontsize=6, loc="upper right")
    ax.grid(alpha=0.2)


# ---------- right column:  winning-pair tally  ----------


# for m in MONKEYS:
#     s = eceo_df[eceo_df.monkey == m]

axB = fig.add_subplot(gs[0, 2])
x = np.arange(len(eceo_pairs)); w = 0.4
axB.bar(x - w / 2, tally["EC"], width=w, color="tab:blue", label="EC")
axB.bar(x + w / 2, tally["EO"], width=w, color="tab:red", label="EO")
axB.set_xticks(x); axB.set_xticklabels(labels, rotation=90, fontsize=6)
axB.set_ylabel("# trials with this pair\nsignificant (percentile < 5)")
axB.set_title("which PC pair carries the map is not fixed", fontsize=8.5)
axB.legend(fontsize=7, frameon=False)
axB.grid(alpha=0.25, axis="y")



# ---------- bottom row: where the map lives in the PCA (all included arrays) ----------
tune = pd.read_csv(os.path.join(TABLE_DIR, "pc_tuning.csv"))
daysim = pd.read_csv(os.path.join(TABLE_DIR, "pc_day_similarity.csv"))
PCS = np.arange(1, MAX_DIM + 1)
PLATEAU_SPAN = (2.5, 7.5)  # PC3-PC7
rng_j = np.random.default_rng(1)

# C: explained variance per PC
axC = fig.add_subplot(gs[1, 0])
axC.axvspan(*PLATEAU_SPAN, color="0.93", lw=0, zorder=0)
axC.boxplot([100 * tune[tune.pc == k].evr for k in PCS], positions=PCS, widths=0.55,
            showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="#86b6ef", edgecolor="0.25", lw=0.6),
            medianprops=dict(color="black", lw=1.2), whiskerprops=dict(lw=0.6),
            capprops=dict(lw=0.6))
axC.set_yscale("log")
axC.set_yticks([1, 2, 5, 10, 20, 50]); axC.set_yticklabels(["1", "2", "5", "10", "20", "50"])
axC.minorticks_off()
axC.set_xticks(PCS); axC.set_xticklabels([f"PC{k}" for k in PCS], fontsize=7)
axC.set_ylabel("variance explained (%)")
axC.text(np.mean(PLATEAU_SPAN), 60, "plateau", ha="center", va="center", fontsize=7, color="0.35")
axC.set_title("PC1 dominates; PC3-7 have\nnearly equal variance", fontsize=8.5)

# D: how often each PC is OP-tuned (label-permutation control)
axD = fig.add_subplot(gs[1, 1])
axD.axvspan(*PLATEAU_SPAN, color="0.93", lw=0, zorder=0)
pooled = tune.groupby("pc").pct_perm.apply(lambda v: 100 * np.mean(v < SIG_THRESH))
axD.bar(PCS, pooled.loc[PCS], width=0.6, color="0.75", edgecolor="0.3", lw=0.5,
        label="all included arrays", zorder=2)
for (m, a), g in tune.groupby(["monkey", "array"]):
    frac = g.groupby("pc").pct_perm.apply(lambda v: 100 * np.mean(v < SIG_THRESH))
    axD.scatter(PCS + rng_j.uniform(-0.18, 0.18, len(PCS)), frac.loc[PCS], s=11,
                color=MONKEY_COLORS[m], edgecolor="0.2", lw=0.3, zorder=3)
axD.set_xticks(PCS); axD.set_xticklabels([f"PC{k}" for k in PCS], fontsize=7)
axD.set_ylim(0, 105)
axD.set_ylabel("% trials in which the PC\nis tuned to the OP map")
axD.set_title("the map lives in the plateau,\nnot in PC1-2", fontsize=8.5)
from matplotlib.lines import Line2D
axD.legend(handles=[Line2D([], [], marker="s", ls="", color="0.75", markeredgecolor="0.3",
                           markersize=6, label="pooled")] +
                  [Line2D([], [], marker="o", ls="", color=MONKEY_COLORS[m], markeredgecolor="0.2",
                          markersize=4.5, label=f"monkey {m} array") for m in MONKEYS],
           fontsize=6, frameon=False, loc="upper left")

# E: day-to-day similarity of single PCs vs the PC3-7 subspace
axE = fig.add_subplot(gs[1, 2])
cats = ["pc1", "pc2", "pc3_7_single", "pc3_7_subspace"]
cat_labels = ["PC1", "PC2", "PC3-7\n(each)", "PC3-7\nsubspace"]
per_arr = daysim.groupby(["monkey", "array"])[cats + ["rand_single", "rand_subspace"]].mean()
xc = np.arange(len(cats))
for (m, a), r in per_arr.iterrows():
    axE.plot(xc, r[cats].values, color=MONKEY_COLORS[m], lw=0.9, alpha=0.8,
             marker="o", ms=3.5, markeredgecolor="0.2", markeredgewidth=0.3, zorder=3)
# end labels, nudged apart where values nearly coincide
ends = per_arr[cats[-1]].sort_values()
label_y, prev = {}, -np.inf
for key, v in ends.items():
    prev = max(v, prev + 0.045); label_y[key] = prev
for (m, a), y in label_y.items():
    axE.text(xc[-1] + 0.12, y, f"{m}{a}", fontsize=5.5, va="center", color="0.25")
rand_single, rand_sub = per_arr.rand_single.mean(), per_arr.rand_subspace.mean()
for k in range(3):
    axE.hlines(rand_single, k - 0.25, k + 0.25, color="0.45", ls="--", lw=1.0, zorder=2)
axE.hlines(rand_sub, 3 - 0.25, 3 + 0.25, color="0.45", ls="--", lw=1.0, zorder=2)
axE.text(0.0, rand_single + 0.03, "random", fontsize=6, color="0.4", ha="center")
axE.set_xticks(xc); axE.set_xticklabels(cat_labels, fontsize=7)
axE.set_xlim(-0.4, len(cats) - 0.3)
axE.set_ylim(0, 1.02)
axE.set_ylabel("similarity between days\n(squared cosine)")
axE.set_title("individual PC3-7 change across days,\nthe subspace they span persists",
              fontsize=8.5)
axE.grid(alpha=0.2, axis="y")


# fig.suptitle("Day-to-day reproducibility of the map-carrying PC pairs\n"
#              "(arrays with $\\geq$25 oriented channels, condition = all, bin = "
#              f"{MAIN_BIN} ms, {NORM})",
#              fontsize=9.5, y=0.995)



def quadrant_label(cell, label, dy=0.045):
    bbox = cell.get_position(fig)
    fig.text(bbox.x0 - 0.02, bbox.y1 + dy, label, fontsize=16,
              fontweight="bold", va="bottom", ha="right")


quadrant_label(gs[0, 0], "A", dy=0.025)   # per-pair profiles across days
quadrant_label(gs[0, 2], "B", dy=0.03)    # winning-pair tally EC vs EO
quadrant_label(gs[1, 0], "C", dy=0.04)    # variance per PC
quadrant_label(gs[1, 1], "D", dy=0.04)    # OP tuning per PC
quadrant_label(gs[1, 2], "E", dy=0.04)    # cross-day similarity


save_fig(fig, "fig4_day_stability")
plt.close(fig)
