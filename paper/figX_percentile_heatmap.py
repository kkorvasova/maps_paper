"""
Alternative (unused) version of the per-array percentile figure: blue/red
heatmap. Kept for reference; the trial-level dot plot in Fig. 1F replaces it.
Min-over-pairs percentile grid (array x condition) for each monkey, averaged
across recording days, at the main FR bin and L2 norm.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import lib
from lib import MONKEYS, TABLE_DIR, SIG_THRESH, save_fig, set_pub_style

set_pub_style()

cellsdf = pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv"))
NORM = "L2"
MAIN_BIN = int(cellsdf.bin_ms.iloc[0])
sub_all = cellsdf[cellsdf.norm == NORM]


def grid_for(monkey):
    sub = sub_all[sub_all.monkey == monkey]
    conds = [c for c in ["all", "EC", "EO"] if c in sub["condition"].unique()]
    arrays = sorted(sub["array"].unique())
    grid = np.full((len(arrays), len(conds)), np.nan)
    for i, arr in enumerate(arrays):
        for j, cond in enumerate(conds):
            v = sub[(sub.array == arr) & (sub.condition == cond)]["percentile"]
            if len(v):
                grid[i, j] = v.mean()  # average across days
    return arrays, conds, grid


fig, axes = plt.subplots(1, len(MONKEYS), figsize=(7.2, 4.4),
                          gridspec_kw=dict(wspace=0.55))
cnorm = TwoSlopeNorm(vmin=0, vcenter=SIG_THRESH, vmax=100)
im = None
for ax, monkey in zip(axes, MONKEYS):
    arrays, conds, grid = grid_for(monkey)
    im = ax.imshow(grid, cmap="bwr", norm=cnorm, aspect="auto")
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
    ax.set_yticks(range(len(arrays))); ax.set_yticklabels(arrays)
    ax.set_xlabel("condition")
    if ax is axes[0]:
        ax.set_ylabel("array")
    ax.set_title(f"monkey {monkey}", fontsize=9)
    for i in range(len(arrays)):
        for j in range(len(conds)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if (v < 2 or v > 70) else "black")

cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03,
                     ticks=[0, SIG_THRESH, 25, 50, 100])
cbar.set_label(f"percentile\nblue < {SIG_THRESH:.0f}% beats control",
               fontsize=7)
cbar.ax.tick_params(labelsize=6.5)
fig.suptitle(f"Min-over-pairs test, averaged across recording days  "
             f"(bin = {MAIN_BIN} ms, {NORM})", fontsize=10, y=0.99)

save_fig(fig, "figX_percentile_heatmap")
plt.close(fig)
