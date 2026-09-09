"""
Figure 3 — a single estimability score per array, and why some arrays
are estimated well and others aren't (oriented-channel count).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lib
from lib import MONKEYS, MONKEY_COLORS, TABLE_DIR, SIG_THRESH, save_fig, set_pub_style

set_pub_style()

rec = pd.read_csv(os.path.join(TABLE_DIR, "estimability_per_array.csv"))
cellsdf = pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv"))
NORM = "L2"
sub = cellsdf[cellsdf.norm == NORM]

rec["label"] = rec["monkey"] + rec["array"].astype(str)
order = []
for m in MONKEYS:
    order += rec[rec.monkey == m].sort_values("frac_sig", ascending=False)["label"].tolist()
rec = rec.set_index("label").loc[order].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                          gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.32))

# --- Panel A: estimability score per array ---
axA = axes[0]
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
axB = axes[1]
for m in MONKEYS:
    s = sub[sub.monkey == m]
    axB.scatter(s["n_oriented"], s["percentile"], s=14, alpha=0.65,
                color=MONKEY_COLORS.get(m, "gray"), edgecolor="k", linewidth=0.2,
                label=m)
axB.axhline(SIG_THRESH, color="crimson", ls="--", lw=1.0, label=f"{SIG_THRESH:.0f}% threshold")
axB.set_xlabel("# oriented channels in array")
axB.set_ylabel(f"{NORM} percentile in null (%)")
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

for ax, lab in zip(axes, "AB"):
    ax.text(-0.16, 1.10, lab, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")

save_fig(fig, "fig3_estimability")
plt.close(fig)
