"""
compute_pc_structure.py
========================
Where in the PCA does the OP map live, and why does the map-carrying PC pair
change across days? (included arrays, main bin)

1. Per cell and PC: explained variance and OP tuning of the PC, i.e. R^2 of
   the PC's channel scores regressed on [cos 2θ, sin 2θ] of the measured OP,
   tested against the label-permutation control (500 permuted maps).
   -> tables/pc_tuning.csv
2. Cross-day similarity of the PCA in channel space, per array / condition /
   day pair: squared cosine between same-numbered PCs (PC1, PC2, each of
   PC3-7) and the mean squared cosine of the principal angles between the
   PC3-7 subspaces, with random-subspace baselines of matching dimension.
   -> tables/pc_day_similarity.csv
Run after compute_summary.py.
"""
import os
import itertools

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles

import lib
from lib import (MONKEYS, DATES, DF_FOLDER, TABLE_DIR, INCLUDED_ARRAYS, MAX_DIM,
                 load_op, load_pca_store)

MAIN_BIN = int(pd.read_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv")).bin_ms.iloc[0])
N_PERM = 500
PLATEAU = slice(2, 7)  # PC3-PC7
rng = np.random.default_rng(0)


def op_r2(score, ops):
    """R^2 of one PC score [n] on [1, cos2θ, sin2θ] for many OP maps [P, n]."""
    out = np.empty(len(ops))
    for i, op in enumerate(ops):
        X = np.c_[np.ones(len(op)), np.cos(np.deg2rad(2 * op)), np.sin(np.deg2rad(2 * op))]
        beta, *_ = np.linalg.lstsq(X, score, rcond=None)
        out[i] = 1 - (score - X @ beta).var() / score.var()
    return out


def unit_cols(S):
    S = S - S.mean(axis=0)
    return S / np.linalg.norm(S, axis=0)


def subspace_overlap(A, B):
    """Mean squared cosine of the principal angles between span(A) and span(B)."""
    return float(np.mean(np.cos(subspace_angles(A, B)) ** 2))


def random_overlap(n, k, reps=200):
    return float(np.mean([subspace_overlap(np.linalg.qr(rng.normal(size=(n, k)))[0],
                                           np.linalg.qr(rng.normal(size=(n, k)))[0])
                          for _ in range(reps)]))


pca = load_pca_store()

# 1. per-PC variance and OP tuning
tune_rows = []
for m in MONKEYS:
    for a in INCLUDED_ARRAYS[m]:
        perms = np.load(os.path.join(DF_FOLDER, "permutations",
                                     f"perm_maps_monkey_{m}_array{a}.npy"))
        op_full = load_op(m, a)
        for d in DATES[m]["RS"]:
            for c in ["all", "EC", "EO"]:
                st = pca.get((m, str(d), a, c, MAIN_BIN))
                if st is None:
                    continue
                oi = st["oriented_idx"]
                real, null_maps = op_full[oi][None, :], perms[:N_PERM, oi]
                for pc in range(min(MAX_DIM, st["scores"].shape[1])):
                    score = st["scores"][:, pc]
                    r2 = op_r2(score, real)[0]
                    null = op_r2(score, null_maps)
                    tune_rows.append(dict(monkey=m, array=a, date=d, condition=c, pc=pc + 1,
                                          evr=float(st["evr"][pc]), r2=float(r2),
                                          pct_perm=float(100 * np.mean(null >= r2))))
tune = pd.DataFrame(tune_rows)
tune.to_csv(os.path.join(TABLE_DIR, "pc_tuning.csv"), index=False)
print("median explained variance per PC:", tune.groupby("pc").evr.median().round(3).to_dict())
print("fraction of trials OP-tuned per PC:",
      tune.groupby("pc").pct_perm.apply(lambda s: float(np.mean(s < 5))).round(2).to_dict())

# 2. cross-day similarity
sim_rows = []
for m in MONKEYS:
    for a in INCLUDED_ARRAYS[m]:
        for c in ["all", "EC", "EO"]:
            days = [d for d in DATES[m]["RS"] if (m, str(d), a, c, MAIN_BIN) in pca]
            for di, dj in itertools.combinations(days, 2):
                A = unit_cols(pca[(m, str(di), a, c, MAIN_BIN)]["scores"])
                B = unit_cols(pca[(m, str(dj), a, c, MAIN_BIN)]["scores"])
                n = A.shape[0]
                cos2 = (A * B).sum(axis=0) ** 2
                sim_rows.append(dict(
                    monkey=m, array=a, condition=c, day_i=di, day_j=dj, n_oriented=n,
                    pc1=cos2[0], pc2=cos2[1], pc3_7_single=float(cos2[PLATEAU].mean()),
                    pc3_7_subspace=subspace_overlap(A[:, PLATEAU], B[:, PLATEAU]),
                    rand_single=random_overlap(n, 1), rand_subspace=random_overlap(n, 5)))
sim = pd.DataFrame(sim_rows)
sim.to_csv(os.path.join(TABLE_DIR, "pc_day_similarity.csv"), index=False)
print(sim.groupby(["monkey", "array"])[["pc1", "pc2", "pc3_7_single", "pc3_7_subspace",
                                        "rand_subspace"]].mean().round(2))
