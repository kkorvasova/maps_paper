"""
paper/lib.py
============
Shared data loading, analysis helpers, and matplotlib style for the
maps_paper publication figures.

Reuses the exact statistical logic already validated in the project's
notebooks (plot_binsweep.ipynb, explore_pca_pairs_full.ipynb) -- this
module does not change the method, only packages it for clean,
reproducible figure generation.

Local data note
----------------
The cluster paths in params_binsweep.yml (op_map_folder, df_folder) are
not reachable from this machine. The dataframes/ folder in this repo is
a local copy of the merged results, and
`~/work/Eduardo_Fernandez/new_method/monkeys_OP/` is a local copy of the
real per-channel OP maps (`final_OP_maps/dataframes/` on the cluster) --
verified row-for-row against the oriented-channel masks stored in
pca_scores_binsweep.pkl (1512/1512 cells match exactly).
"""

import os
import sys
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
DF_FOLDER = os.path.join(PROJECT_ROOT, "dataframes")
OP_MAP_FOLDER = os.path.expanduser(
    "~/work/Eduardo_Fernandez/new_method/monkeys_OP"
)
# Receptive fields and schematic array positions (monkey L only; from the
# Chen et al. V1_V4_1024_electrode_resting_state_data metadata)
L_METADATA_FOLDER = os.path.expanduser(
    "~/work/Eduardo_Fernandez/data/Roelfsema_data/"
    "V1_V4_1024_electrode_resting_state_data/metadata"
)
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TABLE_DIR = os.path.join(PAPER_DIR, "tables")

sys.path.insert(0, CODE_DIR)
from functions_maps import (  # noqa: E402
    load_op_map, polar_angles_from_centroid, best_rotation_to_op,
    apply_geometric_map,
)

with open(os.path.join(CODE_DIR, "params_binsweep.yml")) as f:
    P = yaml.safe_load(f)

MONKEYS = P["monkeys"]
V1_ARRAYS = P["v1_arrays"]
DATES = P["dates"]
BIN_SIZES = P["method"]["bin_sizes"]
LAYOUTS = P["layout"]
N_CH = P["method"]["n_channels"]
SEL_MIN = P["channel_selection"]["selectivity_min"]
JUMP_MAX = P["channel_selection"]["num_jump_max"]
MONKEY_COLORS = P["colors_monkeys"]
MAX_DIM = P["method"]["max_dim"]

SIG_THRESH = 5.0  # percentile threshold used throughout ("beats control")

# Array inclusion: only arrays with MORE than half of their channels oriented
# (> MIN_ORIENTED) enter the paper analyses (counts from
# code/oriented_channel_counts.csv).
MIN_ORIENTED = N_CH // 2
ORIENTED_COUNTS = pd.read_csv(os.path.join(CODE_DIR, "oriented_channel_counts.csv"))
INCLUDED_ARRAYS = {
    m: sorted(int(a) for a in ORIENTED_COUNTS.loc[
        (ORIENTED_COUNTS.monkey == m) & (ORIENTED_COUNTS.n_oriented > MIN_ORIENTED),
        "array"])
    for m in MONKEYS
}


def is_included(monkey, array):
    return int(array) in INCLUDED_ARRAYS.get(monkey, [])


# ─────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────

_cache = {}


def load_compact():
    if "compact" not in _cache:
        with open(os.path.join(DF_FOLDER, "errors_compact_binsweep.pkl"), "rb") as f:
            _cache["compact"] = pickle.load(f)
    return _cache["compact"]


def load_pca_store():
    if "pca_store" not in _cache:
        with open(os.path.join(DF_FOLDER, "pca_scores_binsweep.pkl"), "rb") as f:
            _cache["pca_store"] = pickle.load(f)
    return _cache["pca_store"]


def load_op(monkey, array):
    key = ("op", monkey, array)
    if key not in _cache:
        _cache[key] = load_op_map(
            monkey, array, OP_MAP_FOLDER,
            selectivity_min=SEL_MIN, num_jump_max=JUMP_MAX, n_channels=N_CH)
    return _cache[key]


def load_rf_L():
    """Per-electrode RF centre / size for monkey L, keyed by Array_ID + Electrode_ID."""
    if "rf_L" not in _cache:
        rf = pd.read_csv(os.path.join(L_METADATA_FOLDER, "receptive_fields",
                                      "combined_L_RF.csv"))
        rf = rf.rename(columns={"RF center X (degrees)": "rf_x",
                                "RF center Y (degrees)": "rf_y",
                                "RF size (degrees)": "rf_size"})
        rf["ecc"] = np.hypot(rf.rf_x, rf.rf_y)
        _cache["rf_L"] = rf[["Array_ID", "Electrode_ID", "rf_x", "rf_y",
                             "rf_size", "ecc", "SNR"]]
    return _cache["rf_L"]


def load_array_positions_L():
    """Schematic (not exact) array positions in the cortex, monkey L."""
    pos = pd.read_csv(os.path.join(L_METADATA_FOLDER, "experimental_setup",
                                   "approximate_array_positions_L.csv"))
    return pos.rename(columns={"x (um)": "x_um", "y (um)": "y_um"}).set_index("Array_ID")


def oriented_electrode_ids(monkey, array):
    """Electrode_IDs of the oriented channels of an array (OP-map row order)."""
    df = pd.read_csv(os.path.join(OP_MAP_FOLDER, monkey, f"OP_prop_OG_array{array}.csv"))
    return df.Electrode_ID.values[load_op(monkey, array) >= 0]


def get_layout(monkey, array):
    k = f"{monkey}_even" if int(array) % 2 == 0 else f"{monkey}_odd"
    return np.array(LAYOUTS[k])


def get_scores(monkey, date, array, condition, bin_ms):
    pca_store = load_pca_store()
    key = (monkey, str(date), int(array), condition, int(bin_ms))
    return pca_store.get(key, None)


# ─────────────────────────────────────────────────────────────────
# PC-pair bookkeeping
# ─────────────────────────────────────────────────────────────────

def pair_label(px, py):
    return f"PC{px + 1} vs PC{py + 1}"


def all_pairs(n_pc):
    return list(combinations(range(n_pc), 2))


def derived_map(scores, oriented_idx, op_oriented, px, py):
    x, y = scores[:, px], scores[:, py]
    phi = polar_angles_from_centroid(x, y)
    rot, flip = best_rotation_to_op(phi, op_oriented)
    geo = apply_geometric_map(phi, rot, flip)
    d = np.full(N_CH, -1.0)
    d[oriented_idx] = geo
    return d


def real_err(real_df, monkey, date, array, condition, bin_ms, lab, norm):
    m = ((real_df.monkey == monkey) & (real_df.date == str(date)) &
         (real_df.array == int(array)) & (real_df.condition == condition) &
         (real_df.bin_ms == int(bin_ms)) & (real_df.pc_pair == lab) &
         (real_df.norm == norm))
    v = real_df[m]["real_error"]
    return float(v.iloc[0]) if len(v) else np.nan


def ctrl_arr(ctrl_vec, monkey, date, array, condition, bin_ms, lab, norm):
    k = (monkey, str(date), int(array), condition, int(bin_ms), lab, norm)
    return ctrl_vec.get(k, np.array([]))


def per_pair_percentile(real_df, ctrl_vec, monkey, date, array, condition,
                         bin_ms, n_pc, norm):
    """dict {(px,py): (real_error, percentile, ctrl_vec)} for all pairs."""
    out = {}
    for (px, py) in all_pairs(n_pc):
        lab = pair_label(px, py)
        re = real_err(real_df, monkey, date, array, condition, bin_ms, lab, norm)
        cv = ctrl_arr(ctrl_vec, monkey, date, array, condition, bin_ms, lab, norm)
        pct = 100.0 * np.mean(cv <= re) if (len(cv) and not np.isnan(re)) else np.nan
        out[(px, py)] = (re, pct, cv)
    return out


def minpair_percentile(real_df, ctrl_vec, monkey, date, array, condition,
                        bin_ms, n_pc, norm, return_details=False):
    """Strict test: real min-over-pairs vs control min-over-pairs.

    Same logic as explore_pca_pairs_full.ipynb: stack each pair's control
    vector, take the element-wise min across pairs (same control draw index
    used for every pair), and compare the real best-of-pairs error against
    that null.
    """
    pairs = all_pairs(n_pc)
    real_vals = [real_err(real_df, monkey, date, array, condition, bin_ms,
                           pair_label(*p), norm) for p in pairs]
    valid_pairs = [(p, v) for p, v in zip(pairs, real_vals) if not np.isnan(v)]
    if not valid_pairs:
        return (np.nan, np.nan, None) if return_details else np.nan
    real_min = min(v for _, v in valid_pairs)
    best_pair = min(valid_pairs, key=lambda pv: pv[1])[0]
    ctrl_stack = []
    for p in pairs:
        cv = ctrl_arr(ctrl_vec, monkey, date, array, condition, bin_ms,
                       pair_label(*p), norm)
        if len(cv):
            ctrl_stack.append(cv)
    if not ctrl_stack:
        return (real_min, np.nan, best_pair) if return_details else np.nan
    L = min(len(c) for c in ctrl_stack)
    null_min = np.min(np.vstack([c[:L] for c in ctrl_stack]), axis=0)
    pct = 100.0 * np.mean(null_min <= real_min)
    if return_details:
        return real_min, pct, best_pair, null_min
    return pct


def n_oriented(monkey, array, bin_ms):
    """Oriented-channel count for an array (stable across day/condition)."""
    pca_store = load_pca_store()
    for dt in DATES[monkey]["RS"]:
        for cond in ["all", "EC", "EO"]:
            key = (monkey, str(dt), int(array), cond, int(bin_ms))
            if key in pca_store:
                return len(pca_store[key]["oriented_idx"])
    return 0


# ─────────────────────────────────────────────────────────────────
# Publication style
# ─────────────────────────────────────────────────────────────────

def set_pub_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_fig(fig, name, dpi=300):
    os.makedirs(FIG_DIR, exist_ok=True)
    png = os.path.join(FIG_DIR, f"{name}.png")
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {png}")
    return png, pdf


HSV = None  # set lazily so importing this module doesn't require a display backend


def draw_op_grid(ax, vals, layout, title="", fontsize=8, grid_lines=True,
                  unoriented_color=(0.78, 0.78, 0.78, 1.0)):
    import matplotlib.pyplot as plt
    global HSV
    if HSV is None:
        HSV = plt.get_cmap("hsv")
    uni = np.array(unoriented_color)
    gr = layout.shape
    img = np.ones((*gr, 4))
    for r in range(gr[0]):
        for c in range(gr[1]):
            ch = layout[r, c]
            a = vals[ch] if 0 <= ch < len(vals) else -1
            img[r, c] = uni if a < 0 else HSV((a % 180) / 180.0)
    ax.imshow(img, interpolation="nearest")
    if grid_lines:
        for i in range(gr[1] + 1):
            ax.axvline(i - 0.5, color="white", lw=0.5)
        for i in range(gr[0] + 1):
            ax.axhline(i - 0.5, color="white", lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor("#888888"); sp.set_linewidth(0.6)
