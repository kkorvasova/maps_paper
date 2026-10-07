"""
Supplementary Figure S2 — channel-level relation to SNR, per-array
correlations, and the role of OP coverage / choice of control.
A: example array (L14): reconstruction error and SNR on the 8x8 grid.
B: channel error vs SNR (within-array, monkey L included arrays).
C: per-array Spearman correlation of error with RF size and SNR.
D: OP histograms of an evenly covered and a concentrated array.
E: estimability vs OP concentration, all 29 arrays (permutation control).
F: per-trial percentile, uniform vs label-permutation control.
Needs compute_summary.py and compute_permutation_null.py to have been run.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import panels
from lib import save_fig, set_pub_style

set_pub_style()

fig = plt.figure(figsize=(7.2, 7.6))
gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.0, 1.05], hspace=0.62,
                      left=0.08, right=0.97, top=0.95, bottom=0.07)

# row 1: example maps (error, SNR)
gs_maps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[1.3, 1])
map_axes = panels.draw_channel_maps(fig, gs_maps[0], ex_array=14, fields=("error", "snr"),
                                    wspace=0.55)

# row 2: SNR relation and per-array correlations
gs_mid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.45,
                                          width_ratios=[1.1, 1])
axB = fig.add_subplot(gs_mid[0])
panels.draw_within_array_scatter(axB, "snr", legend=True)
axC = fig.add_subplot(gs_mid[1])
panels.draw_per_array_rho(axC)

# row 3: OP coverage and choice of control
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.5,
                                          width_ratios=[0.8, 1, 1])
rose_axes = panels.draw_op_roses(fig, gs_bot[0])
axE = fig.add_subplot(gs_bot[1])
panels.draw_coverage_scatter(axE)
axF = fig.add_subplot(gs_bot[2])
panels.draw_control_comparison(fig, axF)


def panel_label(ax, label, dx=0.05, dy=0.012):
    bbox = ax.get_position()
    fig.text(bbox.x0 - dx, bbox.y1 + dy, label, fontsize=13, fontweight="bold",
             va="bottom", ha="right")


panel_label(map_axes[0], "A", dx=0.04, dy=0.02)
panel_label(axB, "B", dy=0.05)
panel_label(axC, "C", dx=0.085, dy=0.02)
panel_label(rose_axes[0], "D", dx=0.02, dy=0.03)
panel_label(axE, "E", dy=0.035)
panel_label(axF, "F", dx=0.07, dy=0.02)

save_fig(fig, "figS2_channels_op_coverage")
plt.close(fig)
