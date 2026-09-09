"""
Supplementary Figure S1 — FR bin-size sweep (justifies the main bin) and
L1-vs-L2 percentile agreement (robustness of the norm choice).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lib
from lib import MONKEYS, MONKEY_COLORS, TABLE_DIR, BIN_SIZES, SIG_THRESH, save_fig, set_pub_style

set_pub_style()

bin_sweep = pd.read_csv(os.path.join(TABLE_DIR, "bin_sweep_summary.csv"))
cellsdf = pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv"))
MAIN_BIN = int(cellsdf.bin_ms.iloc[0])

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(wspace=0.4))

axA = axes[0]
for norm, mark in [("L1", "o"), ("L2", "s")]:
    s = bin_sweep[bin_sweep.norm == norm]
    axA.plot(s["bin_ms"], 100 * s["frac_sig"], marker=mark, ms=4, lw=1.3, label=norm)
axA.axvline(MAIN_BIN, color="grey", ls=":", lw=1.0)
axA.set_xscale("log")
axA.set_xticks(BIN_SIZES); axA.set_xticklabels(BIN_SIZES, fontsize=6)
axA.set_xlabel("FR bin size (ms)")
axA.set_ylabel("% cells with percentile < 5%")
axA.set_title(f"main bin = {MAIN_BIN} ms", fontsize=8.5)
axA.legend(fontsize=7, frameon=False)
axA.grid(alpha=0.25)

axB = axes[1]
piv = cellsdf.pivot_table(index=["monkey", "date", "array", "condition"],
                           columns="norm", values="percentile").reset_index()
for m in MONKEYS:
    s = piv[piv.monkey == m]
    axB.scatter(s["L1"], s["L2"], s=18, alpha=0.7, color=MONKEY_COLORS.get(m, "gray"),
                edgecolor="k", linewidth=0.25, label=m)
axB.plot([0, 100], [0, 100], "k--", lw=0.7)
axB.axhline(SIG_THRESH, color="grey", ls=":", lw=0.7)
axB.axvline(SIG_THRESH, color="grey", ls=":", lw=0.7)
axB.set_xlabel("L1 percentile (%)")
axB.set_ylabel("L2 percentile (%)")
axB.set_title(f"L1 vs L2 agreement (bin={MAIN_BIN}ms)", fontsize=8.5)
axB.legend(fontsize=6.5, frameon=False, title="monkey", title_fontsize=6.5)

for ax, lab in zip(axes, "AB"):
    ax.text(-0.20, 1.12, lab, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")

save_fig(fig, "figS1_bin_sweep")
plt.close(fig)
