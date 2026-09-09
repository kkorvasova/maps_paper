"""
Figure 1 — worked example of OP-map estimation from spontaneous activity.
Sketch of the experiment | measured + estimated OP maps (stacked) |
PC-plane projection of the same channels, colored by measured vs. estimated
OP (stacked) | min-over-pairs control null.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import lib
from lib import (load_compact, load_op, get_scores, get_layout, derived_map,
                  minpair_percentile, per_pair_percentile, all_pairs, pair_label,
                  MAX_DIM, save_fig, draw_op_grid, set_pub_style, MONKEYS, MONKEY_COLORS, TABLE_DIR, SIG_THRESH)


set_pub_style()

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


def op_colors(angles_deg):
    return HSV((np.asarray(angles_deg) % 180) / 180.0)


def draw_pc_scatter(ax, angles_deg, title):
    ax.scatter(x_pc, y_pc, c=op_colors(angles_deg), s=30, edgecolor="k",
               linewidth=0.3, zorder=3)
    ax.set_aspect('equal')
    # ax.scatter([cx], [cy], marker="+", s=10, color="black", linewidth=1.4, zorder=4)
    ax.set_xlabel(f"PC{px + 1} score", fontsize=9)
    ax.set_ylabel(f"PC{py + 1} score", fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(labelsize=8)
    ax.axhline(cy, color="grey", lw=0.5, ls=":", zorder=1)
    ax.axvline(cx, color="grey", lw=0.5, ls=":", zorder=1)


fig = plt.figure(figsize=(6, 7))
gs = fig.add_gridspec(3, 2, width_ratios=[1., 1.],
                       height_ratios=[1.,1., 1.], wspace=0.4, hspace=0.6,
                       left=0.035, right=0.985, top=0.95, bottom=0.10)

# --- column 1: sketch of the experiment (spans both rows) ---
ax_sketch = fig.add_subplot(gs[0, 0])
sketch_img = mpimg.imread(lib.PAPER_DIR + "/figures/sketch_monkey.png")
ax_sketch.imshow(sketch_img)
ax_sketch.axis("off")
ax_sketch.set_title("Experiment", fontsize=13, pad=6)

# --- column 2: the two maps, stacked ---
gs_maps = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[1,0], wspace=0.3)
ax_real = fig.add_subplot(gs_maps[0])
draw_op_grid(ax_real, op_full, layout, title="Measured\nOP map", fontsize=11)
ax_real.text(-0.16, 0.5, f"monkey {EX_MONKEY}, array {EX_ARRAY}\n"
             f"{len(oriented_idx)}/{lib.N_CH} oriented channels",
             transform=ax_real.transAxes, ha="right", va="center", fontsize=9,
             rotation=90)

ax_der = fig.add_subplot(gs_maps[1])
draw_op_grid(ax_der, best_map, layout,
             title=f"Estimated map\n({pair_label(*best_pair)})", fontsize=11)

# --- column 3: PC-plane projection of the same channels, colored two ways ---
gs_pc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2,0], wspace=0.5)
ax_sc_real = fig.add_subplot(gs_pc[0])
draw_pc_scatter(ax_sc_real, op_oriented, "PC plane, colored by\nmeasured OP")

ax_sc_est = fig.add_subplot(gs_pc[1])
draw_pc_scatter(ax_sc_est, est_oriented, "PC plane, colored by\nestimated OP")

# shared circular colorbar under all four color-coded panels
sm = mpl.cm.ScalarMappable(cmap="hsv", norm=mpl.colors.Normalize(0, 180))
cbar = fig.colorbar(sm, ax=[ax_real, ax_der],
                     orientation="horizontal", fraction=0.05, pad=0.14,
                     aspect=40, anchor=(0.5, 0.8), location="bottom")
cbar.set_ticks([0, 45, 90, 135, 180])
cbar.set_label("orientation preference (deg)", fontsize=9, labelpad=3)
cbar.ax.tick_params(labelsize=8.5)

# --- column 4: control null (spans both rows) ---
ax_null = fig.add_subplot(gs[0, 1])
ax_null.hist(null_min, bins=40, color="#c6dbef", edgecolor="#4292c6", linewidth=0.5)
p05, p95 = np.percentile(null_min, 5), np.percentile(null_min, 95)
ax_null.axvline(real_min, color="crimson", lw=1.8,
                 label=f"real = {real_min:.1f}°")
ax_null.axvline(p05, color="black", ls="--", lw=1.0, label="null 5th/95th pct")
ax_null.axvline(p95, color="black", ls="--", lw=1.0)
ax_null.set_xlabel(f"best-of-pairs {NORM} error (deg)", fontsize=10.5)
ax_null.set_ylabel("# control maps", fontsize=10.5)
ax_null.set_title(f"percentile = {pct:.2g}%\n(min over all {len(all_pairs(n_pc))} PC pairs)",
                   fontsize=12)
ax_null.legend(fontsize=9, frameon=False, loc="upper left")
ax_null.tick_params(labelsize=9.5)


# Bar plots for success in each array + dependence on orientedness
# -----------------------------------------------------------

rec = pd.read_csv(os.path.join(TABLE_DIR, "estimability_per_array.csv"))
cellsdf = pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv"))
NORM = "L2"
sub = cellsdf[cellsdf.norm == NORM]

rec["label"] = rec["monkey"] + rec["array"].astype(str)
order = []
for m in MONKEYS:
    order += rec[rec.monkey == m].sort_values("frac_sig", ascending=False)["label"].tolist()
rec = rec.set_index("label").loc[order].reset_index()

# --- Panel A: estimability score per array ---
axA = fig.add_subplot(gs[1, 1])
colors = [MONKEY_COLORS.get(m, "gray") for m in rec["monkey"]]
xpos = np.arange(len(rec))
axA.bar(xpos, 100 * rec["frac_sig"], color=colors, edgecolor="k", linewidth=0.3)
axA.set_xticks(xpos)
axA.set_xticklabels(rec["label"], rotation=90, fontsize=5.5)
axA.set_ylabel("% of day x condition cells\nwith percentile < 5%")
axA.set_ylim(0, 100)
axA.axhline(50, color="grey", ls=":", lw=0.8)
n_rec = int((rec.frac_sig >= 0.5).sum())
axA.set_title(f"estimability score per array\n"
              f"({n_rec}/{len(rec)} arrays are estimated in >=50% of cells)", fontsize=8.5)
from matplotlib.patches import Patch
handles = [Patch(facecolor=MONKEY_COLORS[m], edgecolor="k", linewidth=0.3, label=m)
           for m in MONKEYS]
axA.legend(handles=handles, title="monkey", fontsize=6.5, title_fontsize=6.5,
           loc="upper right", frameon=False)

# --- Panel B: percentile vs oriented count ---
axB = fig.add_subplot(gs[2, 1])
for m in MONKEYS:
    s = sub[sub.monkey == m]
    axB.scatter(s["n_oriented"], s["percentile"], s=14, alpha=0.65,
                color=MONKEY_COLORS.get(m, "gray"), edgecolor="k", linewidth=0.2,
                label=m)
axB.axhline(SIG_THRESH, color="crimson", ls="--", lw=1.0, label=f"{SIG_THRESH:.0f}% threshold")
axB.set_xlabel("# oriented channels in array")
axB.set_ylabel(f"{NORM} percentile")
axB.set_title("estimation requires enough\noriented channels", fontsize=8.5)
axB.legend(fontsize=6.5, frameon=False, loc="upper right")

# binned fraction-significant trend, overlaid on a twin axis (noisy per-cell
# scatter vs a clearer trend, as in the exploratory notebook)
edges = np.arange(0, sub["n_oriented"].max() + 6, 8)
cats = pd.cut(sub["n_oriented"], edges, right=False)
frac = sub.groupby(cats, observed=True)["percentile"].apply(lambda s: np.mean(s < SIG_THRESH))
centers = [iv.left + 4 for iv in frac.index]
axB2 = axB.twinx()
axB2.plot(centers, 100 * frac.values, color="black", marker="o", ms=3.5, lw=1.3,
          zorder=6)
axB2.set_ylabel("% cells significant\n(8-channel bins)", fontsize=6.5)
axB2.set_ylim(0, 100)
axB2.tick_params(labelsize=6)


# one label per *quadrant* of the main 2x2 grid, not per individual axes
def quadrant_label(cell, label, dy=0.045):
    bbox = cell.get_position(fig)
    fig.text(bbox.x0 - 0.02, bbox.y1 + dy, label, fontsize=16,
              fontweight="bold", va="bottom", ha="right")

quadrant_label(gs[0, 0], "A", dy=0.025)   # sketch
quadrant_label(gs[0, 1], "D", dy=0.03)   # PC-plane scatters (2-line subtitles)
quadrant_label(gs[1, 0], "B", dy=0.04)   # measured / estimated maps
quadrant_label(gs[2, 0], "C", dy=0.05)   # control null (2-line title)
quadrant_label(gs[1, 1], "E", dy=0.05)   # control null (2-line title)
quadrant_label(gs[2, 1], "F", dy=0.05)   # control null (2-line title)

save_fig(fig, "fig1_example_estimation")
plt.close(fig)
