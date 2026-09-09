"""
compute_errors.py
=================
Main analysis. For each monkey / recording day / V1 array / condition
(all / EC / EO).

Parallelisation
---------------
Run one MONKEY per SLURM array task:

    python compute_errors.py $SLURM_ARRAY_TASK_ID      # 0 -> L, 1 -> N, 2 -> F

With no argument, all monkeys are processed in a single run.

Each task writes its own pickle(s); merge afterwards with:

    python compute_errors.py merge

Steps per cell:

  1. Load tMUA spikes, build the 50 ms firing-rate matrix for the condition.
  2. Restrict to ORIENTED channels, full Pearson correlation matrix, PCA
     (up to max_dim components).  (Option A: PCA on oriented channels only.)
  3. For each PC pair, derive the geometric OP map (optimal rotation/flip
     fit on oriented channels) and compute L1 + L2 error vs the real OP map.
  4. For each of the 1000 pre-generated permuted OP maps, refit the optimal
     rotation/flip in the SAME PC cloud and record its L1 + L2 error.
  5. Save the PCA scores (per array/day/monkey/condition) and a long-form
     error DataFrame.

The PCA scores depend only on activity, not on OP labels, so PCA is
computed once per (monkey, day, array, condition) and reused across the
real map and all permutations.

Outputs (under df_folder/, one set per monkey task):
    pca_scores_{monkey}.pkl    dict[(monkey, date, array, condition)] -> {
                          'scores': [n_oriented, n_pcs],
                          'evr':    explained variance ratios,
                          'mask':   [n_ch] bool oriented,
                          'oriented_idx': [n_oriented] int channel indices,
                      }
    errors_long_{monkey}.pkl   long-form DataFrame (pickle only)

The `merge` command combines the per-monkey files into:
    pca_scores.pkl
    errors_long.pkl

Long-form DataFrame columns:
    monkey, date, array, condition, pc_pair, pc_x, pc_y,
    source ('real' or 'perm'), perm_idx (-1 for real),
    norm ('L1' or 'L2'), error

Run:
    python compute_errors.py 0          # one monkey (SLURM array task)
    python compute_errors.py            # all monkeys, one process
    python compute_errors.py merge      # combine per-monkey outputs
"""

import os
import sys
import time
import glob
import pickle
import numpy as np
import pandas as pd
import yaml

from functions_maps import (
    load_spike_array, load_ec_indicator, load_op_map,
    extract_fr_matrix, compute_correlation_matrix, compute_pca,
    polar_angles_from_centroid, best_rotation_to_op, apply_geometric_map,
    map_errors, pc_pairs, ensure_dir_exists,
)

# ── parameters ────────────────────────────────────────────────────
with open("params_maps.yml") as f:
    P = yaml.safe_load(f)

DATA_FOLDER = P['data_folder']
OP_MAP_FOLDER = P['op_map_folder']
EC_EO_FOLDER = P['ec_eo_folder']
DF_FOLDER = P['df_folder']

MONKEYS = P['monkeys']
V1_ARRAYS = P['v1_arrays']
DATES = P['dates']

BIN_MS = P['method']['bin_ms']
SIG_TYPE = P['method']['sig_type']
CORR_METHOD = P['method']['corr_method']
MAX_DIM = P['method']['max_dim']
N_CH = P['method']['n_channels']
CONDITIONS = P['method']['conditions']

SEL_MIN = P['channel_selection']['selectivity_min']
JUMP_MAX = P['channel_selection']['num_jump_max']

PERM_DIR = os.path.join(DF_FOLDER, 'permutations')

ensure_dir_exists(DF_FOLDER)


def load_perms(monkey, array):
    """Load the pre-generated permuted maps for one array, or None."""
    path = os.path.join(PERM_DIR, f'perm_maps_monkey_{monkey}_array{array}.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)


def errors_for_target(scores, pairs, oriented_idx, target_op_full):
    """
    For one target OP map (real or a permutation), compute L1/L2 errors
    for every PC pair by refitting the optimal rotation/flip in the PC
    cloud.

    target_op_full : [n_ch] OP map (-1 = unoriented). Restricted internally
                     to the oriented channels used in the PCA.

    Returns dict[label] -> (l1, l2)
    """
    op_oriented = target_op_full[oriented_idx]            # [n_oriented]
    valid = op_oriented >= 0                              # all True normally
    out = {}
    for pc_x, pc_y, label in pairs:
        x, y = scores[:, pc_x], scores[:, pc_y]
        phi = polar_angles_from_centroid(x, y)
        # fit rotation/flip on the oriented channels of this target
        best_rot, best_flip = best_rotation_to_op(phi[valid], op_oriented[valid])
        geo = apply_geometric_map(phi, best_rot, best_flip)   # [n_oriented]
        # build full-length derived map for clean comparison
        derived_full = np.full(N_CH, -1.0)
        derived_full[oriented_idx] = geo
        l1, l2 = map_errors(derived_full, target_op_full)
        out[label] = (l1, l2)
    return out


def process_monkey(monkey):
    """Run the full analysis for one monkey and save its per-monkey files."""
    pca_store = {}
    rows = []
    t_start = time.time()

    for date in DATES[monkey]['RS']:

        # EC indicator for this monkey/day (shared across arrays)
        try:
            EC_indic = load_ec_indicator(monkey, date, EC_EO_FOLDER)
        except Exception as exc:
            print(f"[skip day] monkey {monkey} {date}: no EC indicator ({exc})")
            EC_indic = None

        for array in V1_ARRAYS[monkey]:
            t_cell = time.time()

            # ── real OP map (with channel selection) ──────────
            try:
                op_array = load_op_map(
                    monkey, array, OP_MAP_FOLDER,
                    selectivity_min=SEL_MIN, num_jump_max=JUMP_MAX,
                    n_channels=N_CH)
            except Exception as exc:
                print(f"[skip] OP map monkey {monkey} array {array}: {exc}")
                continue

            perms = load_perms(monkey, array)
            if perms is None:
                print(f"[warn] no permutations for monkey {monkey} "
                      f"array {array} -- run generate_permutations.py")

            # ── spikes (loaded once, reused across conditions) ─
            t_load = time.time()
            spike_arr = load_spike_array(
                monkey, array, date, DATA_FOLDER, sig_type=SIG_TYPE)
            load_s = time.time() - t_load
            if spike_arr is None:
                print(f"[skip] spikes monkey {monkey} {date} array {array}")
                continue

            for condition in CONDITIONS:
                if condition in ('EC', 'EO') and EC_indic is None:
                    continue

                fr = extract_fr_matrix(spike_arr, EC_indic, condition, BIN_MS)
                if fr is None:
                    print(f"[skip] {monkey} {date} arr{array} {condition}: "
                          f"too few bins")
                    continue

                C = compute_correlation_matrix(fr, method=CORR_METHOD)
                scores, evr, mask = compute_pca(C, op_array, max_dim=MAX_DIM)
                if scores is None:
                    print(f"[skip] {monkey} {date} arr{array} {condition}: "
                          f"<3 oriented channels")
                    continue

                oriented_idx = np.where(mask)[0]
                pairs = pc_pairs(MAX_DIM, scores.shape[1])

                key = (monkey, date, int(array), condition)
                pca_store[key] = dict(
                    scores=scores, evr=evr, mask=mask,
                    oriented_idx=oriented_idx)

                # ── real-map errors ───────────────────────────
                real_err = errors_for_target(
                    scores, pairs, oriented_idx, op_array)
                for pc_x, pc_y, label in pairs:
                    l1, l2 = real_err[label]
                    for norm, val in (('L1', l1), ('L2', l2)):
                        rows.append(dict(
                            monkey=monkey, date=str(date), array=int(array),
                            condition=condition, pc_pair=label,
                            pc_x=pc_x + 1, pc_y=pc_y + 1,
                            source='real', perm_idx=-1,
                            norm=norm, error=val))

                # ── permutation errors ────────────────────────
                if perms is not None:
                    for p in range(perms.shape[0]):
                        perm_err = errors_for_target(
                            scores, pairs, oriented_idx, perms[p])
                        for pc_x, pc_y, label in pairs:
                            l1, l2 = perm_err[label]
                            for norm, val in (('L1', l1), ('L2', l2)):
                                rows.append(dict(
                                    monkey=monkey, date=str(date),
                                    array=int(array), condition=condition,
                                    pc_pair=label,
                                    pc_x=pc_x + 1, pc_y=pc_y + 1,
                                    source='perm', perm_idx=p,
                                    norm=norm, error=val))

            print(f"done  {monkey} {date} arr{array:2d}  "
                  f"load={load_s:5.1f}s  cell={time.time()-t_cell:5.1f}s  "
                  f"rows so far={len(rows)}")

    # ── save per-monkey files ─────────────────────────────────────
    pca_path = os.path.join(DF_FOLDER, f'pca_scores_{monkey}.pkl')
    with open(pca_path, 'wb') as f:
        pickle.dump(pca_store, f)

    df = pd.DataFrame(rows)
    df_path = os.path.join(DF_FOLDER, f'errors_long_{monkey}.pkl')
    df.to_pickle(df_path)

    print(f"\n[{monkey}] saved {pca_path}  ({len(pca_store)} entries)")
    print(f"[{monkey}] saved {df_path}  ({len(df)} rows)")
    print(f"[{monkey}] total time {time.time()-t_start:.1f}s")


def merge():
    """Combine per-monkey output files into single pca_scores / errors_long."""
    pca_combined = {}
    for path in sorted(glob.glob(os.path.join(DF_FOLDER, 'pca_scores_*.pkl'))):
        with open(path, 'rb') as f:
            pca_combined.update(pickle.load(f))
    out_pca = os.path.join(DF_FOLDER, 'pca_scores.pkl')
    with open(out_pca, 'wb') as f:
        pickle.dump(pca_combined, f)
    print(f"merged PCA scores -> {out_pca}  ({len(pca_combined)} entries)")

    dfs = []
    for path in sorted(glob.glob(os.path.join(DF_FOLDER, 'errors_long_*.pkl'))):
        dfs.append(pd.read_pickle(path))
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        out_df = os.path.join(DF_FOLDER, 'errors_long.pkl')
        df.to_pickle(out_df)
        print(f"merged errors    -> {out_df}  ({len(df)} rows)")
    else:
        print("no per-monkey error files found to merge")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "merge":
        merge()
    elif arg is None:
        # no argument: process every monkey in one process
        for mk in MONKEYS:
            process_monkey(mk)
    else:
        # SLURM array task index -> one monkey
        task = int(arg)
        if task < 0 or task >= len(MONKEYS):
            raise SystemExit(f"task index {task} out of range "
                             f"(0..{len(MONKEYS)-1} for {MONKEYS})")
        process_monkey(MONKEYS[task])
