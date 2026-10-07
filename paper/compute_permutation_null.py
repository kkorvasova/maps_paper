"""
compute_permutation_null.py
============================
Re-test every (monkey, date, array, condition) cell at the main bin with the
LABEL-PERMUTATION control (oriented OP values shuffled among the array's own
oriented channels, so each control map keeps the array's OP histogram)
instead of the uniform-random control.

Purpose: check whether the uniform control favours arrays whose measured OP
histogram is flat (even coverage of orientations). Uses the same rotation
fit, L2 error and min-over-pairs test as the main analysis; the permuted maps
are the pre-generated ones in dataframes/permutations/ (1000 per array).

Runs on ALL arrays (not only the included ones) so the relation to OP
coverage can be assessed across arrays. Writes
    tables/percentile_uniform_vs_perm.csv   (per cell)
    tables/op_coverage_per_array.csv        (per array)
Run after compute_summary.py (needs percentile_by_cell's MAIN_BIN).
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import lib
from lib import (MONKEYS, MAX_DIM, SIG_THRESH, TABLE_DIR, DF_FOLDER, all_pairs,
                 pair_label, minpair_percentile, load_op, n_oriented, is_included)
from functions_maps import polar_angles_from_centroid

MAIN_BIN = int(pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv")).bin_ms.iloc[0])
NORM = "L2"
ROT = np.arange(0, 360, 1.0)


def l2_errors_many(phi, targets):
    """L2 error of the PC-plane angles `phi` [n] against many target maps
    [P, n], refitting rotation/flip per target by min L1 -- identical to
    functions_maps.best_rotation_to_op + map_errors, vectorised over targets."""
    phis = np.stack([phi, (360 - phi) % 360])
    mapped = (((phis[:, None, :] + ROT[None, :, None]) % 360) / 2.0).reshape(-1, len(phi))
    diff = (mapped[None, :, :] - (targets % 180)[:, None, :] + 90) % 180 - 90
    j = np.abs(diff).mean(axis=2).argmin(axis=1)
    best = diff[np.arange(len(targets)), j]
    return np.sqrt((best ** 2).mean(axis=1))


def op_concentration(op_oriented):
    """Resultant length of the doubled OP angles: 0 = orientations evenly
    covered, 1 = all channels share one orientation."""
    return float(np.abs(np.exp(2j * np.deg2rad(op_oriented)).mean()))


def perm_percentile(key):
    m, d, a, c, b = key
    st = lib.get_scores(m, d, a, c, b)
    oi = st["oriented_idx"]
    perms = np.load(os.path.join(DF_FOLDER, "permutations",
                                 f"perm_maps_monkey_{m}_array{a}.npy"))[:, oi]
    real = load_op(m, a)[oi][None, :]
    n_pc = min(MAX_DIM, st["scores"].shape[1])
    real_errs, null_errs = [], []
    for (px, py) in all_pairs(n_pc):
        phi = polar_angles_from_centroid(st["scores"][:, px], st["scores"][:, py])
        real_errs.append(l2_errors_many(phi, real)[0])
        null_errs.append(l2_errors_many(phi, perms))
    real_min = min(real_errs)
    null_min = np.min(np.vstack(null_errs), axis=0)
    return key, float(100.0 * np.mean(null_min <= real_min)), real_min


if __name__ == "__main__":
    compact = lib.load_compact()
    REAL_DF, CTRL_VEC = compact["real"], compact["ctrl"]
    cells = sorted(k for k in lib.load_pca_store() if k[4] == MAIN_BIN)
    print(f"{len(cells)} cells at bin {MAIN_BIN} ms ...")

    with Pool(max(1, os.cpu_count() - 1)) as pool:
        res = pool.map(perm_percentile, cells, chunksize=2)

    rows = []
    for (m, d, a, c, b), pct_perm, real_min in res:
        pct_unif = minpair_percentile(REAL_DF, CTRL_VEC, m, d, a, c, b, MAX_DIM, NORM)
        rows.append(dict(monkey=m, date=d, array=int(a), condition=c, bin_ms=b,
                         real_error=real_min, pct_uniform=pct_unif, pct_perm=pct_perm))
    cells_df = pd.DataFrame(rows)
    cells_df.to_csv(os.path.join(TABLE_DIR, "percentile_uniform_vs_perm.csv"), index=False)

    arr_rows = []
    for (m, a), g in cells_df.groupby(["monkey", "array"]):
        op = load_op(m, a)
        arr_rows.append(dict(
            monkey=m, array=int(a), n_oriented=n_oriented(m, a, MAIN_BIN),
            included=is_included(m, a),
            op_concentration=op_concentration(op[op >= 0]),
            frac_sig_uniform=float(np.mean(g.pct_uniform < SIG_THRESH)),
            frac_sig_perm=float(np.mean(g.pct_perm < SIG_THRESH)),
            n_cells=len(g)))
    arr_df = pd.DataFrame(arr_rows)
    arr_df.to_csv(os.path.join(TABLE_DIR, "op_coverage_per_array.csv"), index=False)
    print(arr_df.round(3).to_string(index=False))
