"""
functions_maps.py
=================
Core routines for the simplified PCA-based orientation-preference (OP)
map inference (maps_paper project).

Pipeline (per monkey / date / array / condition)
------------------------------------------------
  1. Load tMUA spikes (1 ms bins), mask to a condition (all / EC / EO).
  2. Bin firing rate at 50 ms.  NO binarisation, NO upstate detection.
  3. Mask the firing-rate matrix to ORIENTED channels only.
  4. Full Pearson correlation matrix of the (oriented) channels.
  5. PCA on that correlation matrix (up to `max_dim` components).
  6. For each PC pair, derive an OP angle per channel from the polar
     angle in the PC plane, optimally rotated/flipped to match the
     real OP (fit on oriented channels).
  7. Errors (L1 and L2/RMSE) of the derived map vs the real OP map.

The OP-derivation math is taken faithfully from the original
`utah_ec_analysis` module (polar angle -> best rotation/flip).

All time parameters are in ms (== samples, fs = 1000 Hz).
"""

import os
import numpy as np
import pandas as pd
import pickle
from itertools import combinations

import neo
import elephant
import quantities as pq
from scipy.stats import zscore


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────

def load_block(monkey, array, type_rec, type_sig, date, data_folder=''):
    """
    Load preprocessed monkey data (.nix blocks), as produced by the
    snakemake pipeline.

    type_rec : 'RS', 'OG', 'NATIM'
    type_sig : 'LFP', 'RB', 'tMUA', 'spikes', 'spikes_KS4'
    """
    if type_rec == 'NATIM':
        path = (f'{data_folder}/macaque{monkey}_TVSD_{date}/{type_sig}/'
                f'macaque{monkey}_TVSD_{date}_Array{array}_{type_sig}.nix')
    else:
        if type_sig == 'spikes':
            path = (f'{data_folder}/macaque{monkey}_{type_rec}_{date}/{type_sig}/'
                    f'macaque{monkey}_{type_rec}_{date}_Array{array}_spikes_good_units.nix')
        else:
            path = (f'{data_folder}/macaque{monkey}_{type_rec}_{date}/{type_sig}/'
                    f'macaque{monkey}_{type_rec}_{date}_Array{array}_{type_sig}.nix')
    try:
        io = neo.NixIO(path, 'ro')
        block = io.read_block()
        return block
    except Exception:
        print('Could not load:', path)
        return None


def load_spike_array(monkey, array, date, data_folder, sig_type='tMUA'):
    """
    Load tMUA (or other spike) signal and return a binned spike array
    [n_ch, T] at 1 ms resolution.
    """
    spike_block = load_block(monkey, array, type_rec='RS',
                             type_sig=sig_type, date=date,
                             data_folder=data_folder)
    if spike_block is None:
        return None
    bst = elephant.conversion.BinnedSpikeTrain(
        spike_block.segments[0].spiketrains, bin_size=1 * pq.ms)
    return bst.to_array()


def load_ec_indicator(monkey, date, ec_eo_folder):
    """
    Load the EC/EO eyes indicator pickle and return a boolean EC mask [T].
    EC == True means eyes-closed; ~EC means eyes-open.
    """
    file_path = (f'{ec_eo_folder}/'
                 f'eyes_indic_monkey_{monkey}_RS_date_{date}_common_times.pkl')
    with open(file_path, 'rb') as f:
        eyes_dict = pickle.load(f)
    return eyes_dict['EC'].astype(bool)


def load_op_map(monkey, array, op_map_folder,
                selectivity_min=0.15, num_jump_max=2, n_channels=64):
    """
    Load the real measured OP map for one array and apply channel
    selection: a channel becomes UNORIENTED (-1) if its selectivity is
    too low OR it has too many high f0 jumps.

    Returns
    -------
    op_array : ndarray [n_channels]  float, OP in degrees, -1 = unoriented
    """
    df_OP = pd.read_csv(
        f'{op_map_folder}/{monkey}/OP_prop_OG_array{array}.csv')
    op_array = df_OP['pref_OP'].values.copy().astype(float)
    for ch in range(n_channels):
        sel = df_OP.iloc[ch]['selectivity_01']
        jmp = df_OP.iloc[ch]['num_f0_high_jump']
        if sel < selectivity_min or jmp > num_jump_max:
            op_array[ch] = -1
    return op_array


def ensure_dir_exists(dirpath):
    """Create folder on the path if missing. Race-safe for concurrent jobs."""
    os.makedirs(dirpath, exist_ok=True)
    return


# ─────────────────────────────────────────────────────────────────
# FIRING-RATE MATRIX  (50 ms bins, no binarisation, no upstates)
# ─────────────────────────────────────────────────────────────────

def extract_fr_matrix(spike_arr, ec_indic, condition, bin_ms=50):
    """
    Build the firing-rate matrix [n_ch, n_bins] for a condition.

    condition : 'all'  -> use all timepoints
                'EC'   -> eyes-closed bins only
                'EO'   -> eyes-open bins only

    Firing rate is in spikes/ms. No binarisation, no upstate isolation.

    Returns fr [n_ch, n_bins] or None if there are too few bins.
    """
    if condition == 'all':
        spk_cond = spike_arr
    else:
        T = min(spike_arr.shape[1], len(ec_indic))
        spk = spike_arr[:, :T]
        ec = ec_indic[:T].astype(bool)
        mask = ec if condition == 'EC' else ~ec
        spk_cond = spk[:, mask]

    n_ch = spk_cond.shape[0]
    T_cond = spk_cond.shape[1]
    n_bins = T_cond // bin_ms
    if n_bins < 2:
        return None

    fr = (spk_cond[:, :n_bins * bin_ms]
          .reshape(n_ch, n_bins, bin_ms)
          .sum(axis=2) / bin_ms)        # spikes/ms, [n_ch, n_bins]
    return fr


def compute_correlation_matrix(fr, method='pearson'):
    """
    Channel x channel similarity matrix from the firing-rate matrix.

    method : 'pearson' | 'cosine'
    Returns C [n_ch, n_ch].
    """
    n_ch = fr.shape[0]
    method = method.lower()
    if method == 'pearson':
        C = np.corrcoef(fr)
        np.fill_diagonal(C, 1.0)
        C = np.nan_to_num(C, nan=0.0)
    elif method == 'cosine':
        norms = np.linalg.norm(fr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normed = fr / norms
        C = np.clip(normed @ normed.T, -1, 1)
    else:
        raise ValueError(f"Unknown corr method {method!r}.")
    return C


# ─────────────────────────────────────────────────────────────────
# PCA  (oriented channels only -- Option A)
# ─────────────────────────────────────────────────────────────────

def compute_pca(C, angles_deg, max_dim=7):
    """
    PCA where each channel is a point described by its row in C.
    Unoriented channels (angle == -1) are EXCLUDED from C before fitting.

    Parameters
    ----------
    C          : [n_ch, n_ch] correlation matrix (full, all channels)
    angles_deg : [n_ch] real OP angles (-1 = unoriented)
    max_dim    : number of principal components to keep (<= n oriented)

    Returns
    -------
    scores : [n_oriented, n_components]  channel coordinates in PC space
    evr    : explained variance ratios
    mask   : [n_ch] bool, the oriented channels kept (rows of `scores`)
    """
    from sklearn.decomposition import PCA as _PCA

    mask = angles_deg != -1
    C_fit = C[np.ix_(mask, mask)]
    n_keep = mask.sum()
    if n_keep < 3:
        return None, None, mask

    n_comp = min(C_fit.shape[0], max_dim)
    pca = _PCA(n_components=n_comp)
    scores = pca.fit_transform(C_fit)
    evr = pca.explained_variance_ratio_
    return scores, evr, mask


# ─────────────────────────────────────────────────────────────────
# GEOMETRIC OP DERIVATION  (polar angle -> best rotation / flip)
# ─────────────────────────────────────────────────────────────────

def polar_angles_from_centroid(x, y):
    """
    Polar angle (degrees, 0-360) of each 2-D point relative to the
    centre of mass of the cloud.
    """
    cx, cy = x.mean(), y.mean()
    return np.rad2deg(np.arctan2(y - cy, x - cx)) % 360


def circular_diff(mapped, real):
    """
    Wrapped circular difference on [0,180): result in [-90, 90).
    Both inputs in degrees.
    """
    return (mapped - real + 90) % 180 - 90


def best_rotation_to_op(phi_deg, op_deg, rot_step=1.0):
    """
    Find the rotation offset (0-360, step `rot_step`) and optional flip
    that maps polar angles `phi_deg` to orientation (phi/2) so as to best
    match the real OP angles `op_deg`.  Orientation is pi-periodic, so we
    compare modulo 180.

    `phi_deg` and `op_deg` are restricted to the SAME (oriented) channels
    for the fit.

    Returns
    -------
    best_rot  : float, rotation offset (deg)
    best_flip : bool, whether phi was negated before rotation
    """
    op = op_deg % 180
    rotations = np.arange(0, 360, rot_step)

    best_err, best_rot, best_flip = np.inf, 0.0, False
    for flip in (False, True):
        phi_use = (360 - phi_deg) % 360 if flip else phi_deg
        # mapped[r, i] for rotation r, channel i
        mapped = ((phi_use[None, :] + rotations[:, None]) % 360) / 2.0
        diff = (mapped - op[None, :] + 90) % 180 - 90
        errs = np.abs(diff).mean(axis=1)
        j = int(np.argmin(errs))
        if errs[j] < best_err:
            best_err, best_rot, best_flip = errs[j], rotations[j], flip
    return best_rot, best_flip


def apply_geometric_map(phi_deg, best_rot, best_flip):
    """Map polar angles to OP angles [0,180) given rotation & flip."""
    phi_use = (360 - phi_deg) % 360 if best_flip else phi_deg
    return ((phi_use + best_rot) % 360) / 2.0


def derive_op_for_pair(scores, pc_x, pc_y, op_oriented, rot_step=1.0):
    """
    Derive the geometric OP map for one PC pair, fitting rotation/flip on
    the oriented channels (all channels in `scores` ARE oriented here,
    since PCA was run on oriented channels only).

    Parameters
    ----------
    scores      : [n_oriented, n_pcs]
    pc_x, pc_y  : zero-indexed PC indices
    op_oriented : [n_oriented] real OP angles (all >= 0)

    Returns
    -------
    geo_op : [n_oriented] derived OP in [0,180)
    """
    x, y = scores[:, pc_x], scores[:, pc_y]
    phi = polar_angles_from_centroid(x, y)
    best_rot, best_flip = best_rotation_to_op(phi, op_oriented, rot_step)
    return apply_geometric_map(phi, best_rot, best_flip)


def pc_pairs(max_dim, n_components):
    """
    List of (pc_x, pc_y, label) zero-indexed PC pairs up to max_dim,
    bounded by the number of available components.
    """
    n = min(max_dim, n_components)
    return [(i, j, f"PC{i+1} vs PC{j+1}")
            for i, j in combinations(range(n), 2)]


# ─────────────────────────────────────────────────────────────────
# ERROR METRICS  (L1 and L2 / RMSE on circular error)
# ─────────────────────────────────────────────────────────────────

def map_errors(derived_op, real_op):
    """
    L1 (mean abs) and L2 (RMSE) circular error (deg) between two OP maps,
    computed on channels valid (>= 0) in both.

    Returns (l1, l2). NaN if fewer than 2 valid channels.
    """
    valid = (derived_op >= 0) & (real_op >= 0)
    if valid.sum() < 2:
        return np.nan, np.nan
    diff = circular_diff(derived_op[valid], real_op[valid])
    l1 = float(np.abs(diff).mean())
    l2 = float(np.sqrt((diff ** 2).mean()))
    return l1, l2


# ─────────────────────────────────────────────────────────────────
# PERMUTATIONS  (shuffle oriented OP labels, unoriented stay -1)
# ─────────────────────────────────────────────────────────────────

def make_permuted_maps(op_array, n_perm=1000, seed=0):
    """
    Build `n_perm` permuted copies of the real OP map.  Only the ORIENTED
    channels' values are shuffled among themselves; unoriented channels
    keep -1 and their positions are preserved.

    Returns
    -------
    perms : ndarray [n_perm, n_ch]  each row a permuted OP map
    """
    op_array = np.asarray(op_array, dtype=float)
    n_ch = len(op_array)
    oriented_idx = np.where(op_array != -1)[0]
    oriented_vals = op_array[oriented_idx]

    rng = np.random.default_rng(seed)
    perms = np.full((n_perm, n_ch), -1.0)
    for p in range(n_perm):
        shuffled = rng.permutation(oriented_vals)
        perms[p, oriented_idx] = shuffled
    return perms