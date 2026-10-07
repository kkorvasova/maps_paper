"""
Figure 1 — OP-map estimation from spontaneous activity and where it works.
Row 1: A sketch | B measured + estimated OP map | C PC-plane projection |
       D min-over-pairs control null.
Row 2: E estimability per array | F trial-level percentile of every included
       array (day x condition).
Row 3: G RF centres | H channel-level example (error / RF size) |
       I error vs RF size.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import lib
import panels
from lib import (load_compact, load_op, get_scores, get_layout, derived_map,
                  minpair_percentile, per_pair_percentile, all_pairs, pair_label,
                  MAX_DIM, save_fig, draw_op_grid, set_pub_style, MONKEYS, MONKEY_COLORS, TABLE_DIR, SIG_THRESH)


set_pub_style()
# full-page figure drawn at 11 in wide -> printed at ~7.2 in (x0.65); fonts are
# set ~1.5x the default so they print at ~7 pt
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
                     "axes.linewidth": 0.9, "xtick.major.width": 0.9,
                     "ytick.major.width": 0.9})

EX_MONKEY, EX_ARRAY = "L", 16
EX_DATE = lib.DATES["L"]["RS"][1]
EX_COND, EX_BIN, NORM = "all", 100, "L2"

compact = load_compact()
REAL_DF, CTRL_VEC = compact["real"], compact["ctrl"]

st = get_scores(EX_MONKEY, EX_DATE, EX_ARRAY, EX_COND, EX_BIN)
scores, oriented_idx, evr = st["scores"], st["oriented_idx"], st["evr"]
n_pc = min(MAX_DIM, scores.shape[1])
op_full = load_op(EX_MONKEY, EX_ARRAY)
op_oriented = op_full[oriented_idx]
layout = get_layout(EX_MONKEY, EX_ARRAY)

real_min, pct, best_pair, null_min = minpair_percentile(
    REAL_DF, CTRL_VEC, EX_MONKEY, EX_DATE, EX_ARRAY, EX_COND, EX_BIN, n_pc, NORM,
    return_details=True)
best_map = derived_map(scores, oriented_idx, op_oriented, *best_pair)
est_oriented = best_map[oriented_idx]  # estimated OP, same channels as op_oriented

px, py = best_pair
x_pc, y_pc = scores[:, px], scores[:, py]
cx, cy = x_pc.mean(), y_pc.mean()

print(f"example: {EX_MONKEY} {EX_DATE} arr{EX_ARRAY} {EX_COND} bin{EX_BIN} | "
      f"n_oriented={len(oriented_idx)} | best pair={pair_label(*best_pair)} | "
      f"real {NORM} error={real_min:.2f} deg | percentile={pct:.2f}%")

HSV = plt.get_cmap("hsv")
FS = 11  # base font size (prints at ~7 pt after scaling to page width)


def op_colors(angles_deg):
    return HSV((np.asarray(angles_deg) % 180) / 180.0)


def draw_pc_scatter(ax, angles_deg, title, ylabel=True):
    ax.scatter(x_pc, y_pc, c=op_colors(angles_deg), s=34, edgecolor="k",
               linewidth=0.25, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC{px + 1} score", fontsize=FS)
    if ylabel:
        ax.set_ylabel(f"PC{py + 1} score", fontsize=FS, labelpad=1)
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=FS)
    ax.tick_params(labelsize=FS - 1.5)
    ax.axhline(cy, color="grey", lw=0.5, ls=":", zorder=1)
    ax.axvline(cx, color="grey", lw=0.5, ls=":", zorder=1)


fig = plt.figure(figsize=(11, 13.2))
gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.65, 1.], hspace=0.5,
                      left=0.05, right=0.975, top=0.955, bottom=0.065)
# row 1: A-D | row 2: E, F | row 3: G, H, I
# narrow empty column between B and C keeps B's titles clear of C's axis label
row1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], wspace=0.45,
                                        width_ratios=[1.35, 0.95, 0.15, 0.85, 1.2])
row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.15,
                                        width_ratios=[1., 3.2])
row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.4,
                                        width_ratios=[1.75, 0.65, 1.15])
cA, cB, cC, cD = row1[0], row1[1], row1[3], row1[4]
cE, cF = row2[0], row2[1]
cG, cH, cI = row3[0], row3[1], row3[2]

# ---------- row 1 ----------
# A: sketch of the experiment
ax_sketch = fig.add_subplot(cA)
ax_sketch.imshow(mpimg.imread(lib.PAPER_DIR + "/figures/sketch_monkey.png"))
ax_sketch.axis("off")
ax_sketch.set_title("Experiment", fontsize=FS + 1.5, pad=4)

# B: measured and estimated map
gs_maps = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=cB, hspace=0.38)
ax_real = fig.add_subplot(gs_maps[0])
draw_op_grid(ax_real, op_full, layout,
             title=f"measured OP map\n(L{EX_ARRAY}, {len(oriented_idx)}/{lib.N_CH} oriented)",
             fontsize=FS - 0.5)
ax_der = fig.add_subplot(gs_maps[1])
draw_op_grid(ax_der, best_map, layout,
             title=f"estimated map\n({pair_label(*best_pair)})", fontsize=FS - 0.5)
sm = mpl.cm.ScalarMappable(cmap="hsv", norm=mpl.colors.Normalize(0, 180))
cbar = fig.colorbar(sm, ax=[ax_real, ax_der], orientation="horizontal", fraction=0.05,
                    pad=0.04, aspect=18, location="bottom")
cbar.set_ticks([0, 45, 90, 135, 180])
cbar.set_label("orientation preference (deg)", fontsize=FS - 1.5, labelpad=2)
cbar.ax.tick_params(labelsize=FS - 1.5)

# C: PC-plane projection of the same channels, coloured two ways
gs_pc = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=cC, hspace=0.25)
ax_pc0 = fig.add_subplot(gs_pc[0])
draw_pc_scatter(ax_pc0, op_oriented, "measured OP")
ax_pc1 = fig.add_subplot(gs_pc[1], sharex=ax_pc0)
draw_pc_scatter(ax_pc1, est_oriented, "estimated OP")
ax_pc0.set_xlabel(""); ax_pc0.tick_params(labelbottom=False)

# D: control null
ax_null = fig.add_subplot(cD)
counts, _, _ = ax_null.hist(null_min, bins=40, color="#c6dbef", edgecolor="#4292c6",
                            linewidth=0.4)
p05, p95 = np.percentile(null_min, 5), np.percentile(null_min, 95)
top = counts.max() * 1.05  # lines stop below the legend headroom
ax_null.vlines(real_min, 0, top, color="crimson", lw=1.6, label=f"real = {real_min:.1f}°")
ax_null.vlines([p05, p95], 0, top, color="black", ls="--", lw=0.9, label="null 5th/95th pct")
ax_null.set_xlabel(f"best-of-pairs {NORM} error (deg)")
ax_null.set_ylabel("# control maps")
ax_null.set_title(f"percentile = {pct:.2g}%\n(min over all {len(all_pairs(n_pc))} PC pairs)",
                  fontsize=FS + 0.5)
ax_null.set_ylim(0, ax_null.get_ylim()[1] * 1.45)  # headroom for the legend
ax_null.legend(fontsize=FS - 1.5, frameon=False, loc="upper left")

# ---------- row 2 ----------
# E: estimability score per included array
rec = pd.read_csv(os.path.join(TABLE_DIR, "estimability_per_array.csv"))
rec["label"] = rec["monkey"] + rec["array"].astype(str)
order = []
for m in MONKEYS:
    order += rec[rec.monkey == m].sort_values("frac_sig", ascending=False)["label"].tolist()
rec = rec.set_index("label").loc[order].reset_index()
axE = fig.add_subplot(cE)
xpos = np.arange(len(rec))
axE.bar(xpos, 100 * rec["frac_sig"], color=[MONKEY_COLORS[m] for m in rec["monkey"]],
        edgecolor="k", linewidth=0.3)
axE.set_xticks(xpos)
axE.set_xticklabels(rec["label"], rotation=90, fontsize=FS - 1)
axE.set_ylabel("% of day x condition trials\nwith percentile < 5%")
axE.set_ylim(0, 100)
axE.axhline(50, color="grey", ls=":", lw=0.8)
n_rec = int((rec.frac_sig >= 0.5).sum())
axE.set_title(f"estimability per array\n({n_rec}/{len(rec)} estimated in >=50% of trials)",
              fontsize=FS + 0.5)
from matplotlib.patches import Patch
axE.legend(handles=[Patch(facecolor=MONKEY_COLORS[m], edgecolor="k", linewidth=0.3, label=m)
                    for m in MONKEYS],
           title="monkey", fontsize=FS - 1.5, title_fontsize=FS - 1.5, loc="upper right",
           frameon=False)

# ---------- row 3 ----------
# G: RF centres coloured by estimability (monkey L)
allarr = pd.read_csv(os.path.join(TABLE_DIR, "estimability_all_arrays.csv"))
EST_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "est", plt.get_cmap("Blues")(np.linspace(0.15, 1.0, 256)))
est_norm = mpl.colors.Normalize(0, 1)
L_arr = allarr[allarr.monkey == "L"].set_index("array")
rf_L = lib.load_rf_L()
axRF = fig.add_subplot(cG)
for arr, r in L_arr[L_arr.included].iterrows():
    rfa = rf_L[rf_L.Electrode_ID.isin(lib.oriented_electrode_ids("L", arr))]
    axRF.scatter(rfa.rf_x, rfa.rf_y, s=13, color=EST_CMAP(est_norm(r.frac_sig)),
                edgecolor="0.25", linewidth=0.2, zorder=3)
for arr, r in L_arr[L_arr.included].iterrows():
    axRF.text(r.rf_x, r.rf_y, f"L{arr}", fontsize=FS - 1, fontweight="bold", ha="center",
             va="center", zorder=4,
             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
axRF.axhline(0, color="grey", lw=0.5); axRF.axvline(0, color="grey", lw=0.5)
axRF.set_aspect("equal")
axRF.set_xlabel("RF azimuth (deg)")
axRF.set_ylabel("RF elevation (deg)")
axRF.set_title("RF centres of oriented channels\n(monkey L)", fontsize=FS + 0.5)
cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=EST_CMAP, norm=est_norm), ax=axRF,
                  orientation="vertical", fraction=0.05, pad=0.03)
cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0%", "50%", "100%"])
cb.set_label("% trials significant", fontsize=FS - 1.5)
cb.ax.tick_params(labelsize=FS - 2)

# F (row 2): every trial of every included array
axDot = fig.add_subplot(cF)
panels.draw_trial_dotplot(axDot, marker_size=60, fontsize=FS - 0.5)
axDot.set_ylabel("array")

# H, I (row 3): channel level
map_axes = panels.draw_channel_maps(fig, cH, ex_array=14, fields=("error", "rf_size"),
                                    title_fs=FS, vertical=True, hspace=0.4)
axI = fig.add_subplot(cI)
panels.draw_within_array_scatter(axI, "rf_size", legend=True, fontsize=FS - 0.5,
                                  marker_size=24)


def panel_label(ref, label, dx=0.03, dy=0.012):
    """Bold letter at the top-left of an axes or subplot spec (figure coords)."""
    bbox = ref.get_position(fig) if hasattr(ref, "get_gridspec") else ref.get_position()
    fig.text(bbox.x0 - dx, bbox.y1 + dy, label, fontsize=19, fontweight="bold",
             va="bottom", ha="right")


panel_label(cA, "A", dx=0.0, dy=0.0)
panel_label(cB, "B", dx=0.015, dy=0.0)
panel_label(cC, "C", dx=0.045, dy=0.0)
panel_label(cD, "D", dy=0.0)
panel_label(cE, "E", dx=0.04, dy=0.02)
panel_label(axDot, "F", dx=0.025, dy=0.02)
panel_label(cG, "G", dy=0.02)
panel_label(map_axes[0], "H", dx=0.035, dy=0.02)
panel_label(axI, "I", dx=0.045, dy=0.02)

save_fig(fig, "fig1_example_estimation")
plt.close(fig)
