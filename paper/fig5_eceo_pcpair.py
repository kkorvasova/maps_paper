"""
Figure 5 — eyes-closed vs eyes-open estimation, and the shifting map-carrying
PC pair.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lib
from lib import (MONKEYS, MONKEY_COLORS, TABLE_DIR, SIG_THRESH, MAX_DIM,
                  save_fig, set_pub_style, per_pair_percentile, all_pairs,
                  pair_label, minpair_percentile)

set_pub_style()

compact = lib.load_compact()
REAL_DF, CTRL_VEC = compact["real"], compact["ctrl"]
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

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                          gridspec_kw=dict(width_ratios=[1.0, 1.5], wspace=0.35))

# --- Panel A: EC vs EO percentile scatter ---
axA = axes[0]
for m in MONKEYS:
    s = eceo_df[eceo_df.monkey == m]
    axA.scatter(s["pct_EC"], s["pct_EO"], s=26, alpha=0.8,
                color=MONKEY_COLORS.get(m, "gray"), edgecolor="k", linewidth=0.3,
                label=m)
axA.plot([0, 100], [0, 100], "k--", lw=0.7)
axA.axhline(SIG_THRESH, color="grey", ls=":", lw=0.8)
axA.axvline(SIG_THRESH, color="grey", ls=":", lw=0.8)
axA.set_xlabel("EC percentile (%)")
axA.set_ylabel("EO percentile (%)")
n_both = int(((eceo_df.pct_EC < SIG_THRESH) & (eceo_df.pct_EO < SIG_THRESH)).sum())
axA.set_title(f"EC vs EO estimation\n({n_both}/{len(eceo_df)} cells significant in both)",
              fontsize=8.5)
axA.legend(fontsize=6.5, frameon=True, framealpha=0.85, edgecolor="none",
           loc="lower right", title="monkey", title_fontsize=6.5)

# --- Panel B: winning-pair tally ---
axB = axes[1]
x = np.arange(len(eceo_pairs)); w = 0.4
axB.bar(x - w / 2, tally["EC"], width=w, color="tab:blue", label="EC")
axB.bar(x + w / 2, tally["EO"], width=w, color="tab:red", label="EO")
axB.set_xticks(x); axB.set_xticklabels(labels, rotation=90, fontsize=6)
axB.set_ylabel("# cells with this pair\nsignificant (pct < 5%)")
axB.set_title("which PC pair carries the map is not fixed", fontsize=8.5)
axB.legend(fontsize=7, frameon=False)
axB.grid(alpha=0.25, axis="y")

for ax, lab in zip(axes, "AB"):
    ax.text(-0.18, 1.12, lab, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")

save_fig(fig, "fig5_eceo_pcpair")
plt.close(fig)

print(f"qualifying cells (sig in EC or EO): {len(qual)}")
print("EC tally:", dict(zip(labels, tally['EC'].astype(int))))
print("EO tally:", dict(zip(labels, tally['EO'].astype(int))))
