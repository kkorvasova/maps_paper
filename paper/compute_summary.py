"""
compute_summary.py
===================
Compute all summary tables/numbers used by the paper figures and the
manuscript text. Run once; writes CSVs to paper/tables/ and a JSON of
headline numbers to paper/tables/headline_numbers.json.
"""
import os
import json
import itertools
import numpy as np
import pandas as pd

import lib
from lib import (MONKEYS, V1_ARRAYS, DATES, BIN_SIZES, MAX_DIM, SIG_THRESH,
                  TABLE_DIR, minpair_percentile, n_oriented, per_pair_percentile,
                  all_pairs, pair_label)

os.makedirs(TABLE_DIR, exist_ok=True)
compact = lib.load_compact()
REAL_DF, CTRL_VEC = compact["real"], compact["ctrl"]
pca_store = lib.load_pca_store()

CELLS = sorted({(m, d, int(a), c, int(b))
                for (m, d, a, c, b) in pca_store.keys()})
print(f"total (monkey,date,array,condition,bin) cells: {len(CELLS)}")


# ─────────────────────────────────────────────────────────────────
# 1. Bin-size sweep -> pick the main bin
# ─────────────────────────────────────────────────────────────────
print("\n[1/6] bin-size sweep ...")
rows = []
for bm in BIN_SIZES:
    for norm in ["L1", "L2"]:
        pcts = []
        for (m, d, a, c, b) in CELLS:
            if b != bm:
                continue
            p = minpair_percentile(REAL_DF, CTRL_VEC, m, d, a, c, bm, MAX_DIM, norm)
            if not np.isnan(p):
                pcts.append(p)
        pcts = np.array(pcts)
        rows.append(dict(bin_ms=bm, norm=norm,
                          frac_sig=float(np.mean(pcts < SIG_THRESH)) if len(pcts) else np.nan,
                          median_pct=float(np.median(pcts)) if len(pcts) else np.nan,
                          n=len(pcts)))
bin_sweep = pd.DataFrame(rows)
bin_sweep.to_csv(os.path.join(TABLE_DIR, "bin_sweep_summary.csv"), index=False)
print(bin_sweep)

MAIN_BIN = int(bin_sweep.loc[bin_sweep.norm == "L2"].sort_values("frac_sig", ascending=False).iloc[0].bin_ms)
print("MAIN_BIN (best L2 frac_sig) =", MAIN_BIN)


# ─────────────────────────────────────────────────────────────────
# 2. Per-cell percentile table at MAIN_BIN (both norms)
# ─────────────────────────────────────────────────────────────────
print("\n[2/6] per-cell percentile table at MAIN_BIN ...")
rows = []
for (m, d, a, c, b) in CELLS:
    if b != MAIN_BIN:
        continue
    for norm in ["L1", "L2"]:
        real_min, pct, best_pair, null_min = minpair_percentile(
            REAL_DF, CTRL_VEC, m, d, a, c, MAIN_BIN, MAX_DIM, norm, return_details=True)
        rows.append(dict(monkey=m, date=d, array=a, condition=c, bin_ms=MAIN_BIN,
                          norm=norm, real_error=real_min, percentile=pct,
                          best_pair=lib.pair_label(*best_pair) if best_pair else None,
                          n_oriented=n_oriented(m, a, MAIN_BIN)))
cellsdf = pd.DataFrame(rows)
cellsdf.to_csv(os.path.join(TABLE_DIR, "percentile_by_cell.csv"), index=False)
print(f"  {len(cellsdf)} rows")


# ─────────────────────────────────────────────────────────────────
# 3. Estimability score per array (pooled across day & condition), L2
# ─────────────────────────────────────────────────────────────────
print("\n[3/6] estimability score per array ...")
NORM_MAIN = "L2"
sub = cellsdf[cellsdf.norm == NORM_MAIN]
rec = (sub.groupby(["monkey", "array"])
       .agg(frac_sig=("percentile", lambda s: float(np.mean(s < SIG_THRESH))),
            median_pct=("percentile", "median"),
            n_cells=("percentile", "size"),
            n_oriented=("n_oriented", "max"))
       .reset_index())
rec = rec.sort_values(["monkey", "frac_sig"], ascending=[True, False])
rec.to_csv(os.path.join(TABLE_DIR, "estimability_per_array.csv"), index=False)
print(rec.to_string(index=False))

n_estimable = int((rec.frac_sig >= 0.5).sum())  # majority of its cells significant
n_arrays = len(rec)
print(f"arrays with >=50% of cells significant ({NORM_MAIN}): {n_estimable} / {n_arrays}")


# ─────────────────────────────────────────────────────────────────
# 4. Oriented-channel-count cutoff
# ─────────────────────────────────────────────────────────────────
print("\n[4/6] oriented-count cutoff ...")
sig_counts = sub[sub.percentile < SIG_THRESH]["n_oriented"]
min_sig = int(sig_counts.min()) if len(sig_counts) else None
p10_sig = int(np.percentile(sig_counts, 10)) if len(sig_counts) else None
print(f"smallest oriented count with a significant cell ({NORM_MAIN}): {min_sig}")
print(f"10th percentile of oriented count among significant cells: {p10_sig}")


# ─────────────────────────────────────────────────────────────────
# 5. Day-to-day stability (arrays with >=25 oriented channels, >1 day)
# ─────────────────────────────────────────────────────────────────
print("\n[5/6] day-to-day stability ...")
STAB_NPC = MAX_DIM
MIN_ORIENTED_STAB = 25
TOPK = 3
stab_pairs = all_pairs(STAB_NPC)


def day_error_vector(monkey, dt, arr, cond, bin_ms, norm):
    pp = per_pair_percentile(REAL_DF, CTRL_VEC, monkey, dt, arr, cond, bin_ms, STAB_NPC, norm)
    return np.array([pp[p][0] for p in stab_pairs])


qualifying = []
for monkey in MONKEYS:
    arrays = sorted({int(a) for (m, d, a, c, b) in pca_store if m == monkey and b == MAIN_BIN})
    for arr in arrays:
        if n_oriented(monkey, arr, MAIN_BIN) < MIN_ORIENTED_STAB:
            continue
        days = [dt for dt in DATES[monkey]["RS"]
                if lib.get_scores(monkey, dt, arr, "all", MAIN_BIN) is not None]
        if len(days) > 1:
            qualifying.append((monkey, arr, days))
print("qualifying arrays (>=25 oriented, >1 day):", [(m, a) for (m, a, _) in qualifying])

pool_rows = []
for (monkey, arr, days) in qualifying:
    evecs = {dt: day_error_vector(monkey, dt, arr, "all", MAIN_BIN, NORM_MAIN) for dt in days}
    tops = {dt: set(np.argsort(evecs[dt])[:TOPK]) for dt in days}
    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            di, dj = days[i], days[j]
            v = ~np.isnan(evecs[di]) & ~np.isnan(evecs[dj])
            l2 = float(np.sqrt(np.mean((evecs[di][v] - evecs[dj][v]) ** 2)))
            inter = len(tops[di] & tops[dj]); union = len(tops[di] | tops[dj])
            jac = inter / union if union else np.nan
            pool_rows.append(dict(monkey=monkey, array=arr, day_i=di, day_j=dj,
                                   day_pair=f"{i + 1}-{j + 1}", l2=l2, jaccard=jac))
day_stability = pd.DataFrame(pool_rows)
day_stability.to_csv(os.path.join(TABLE_DIR, "day_stability.csv"), index=False)
print(f"  {len(day_stability)} day-pair rows across {len(qualifying)} arrays")

# save the qualifying-array list too (needed to rebuild the figure)
with open(os.path.join(TABLE_DIR, "qualifying_stability_arrays.json"), "w") as f:
    json.dump([(m, a, days) for (m, a, days) in qualifying], f, indent=1)


# ─────────────────────────────────────────────────────────────────
# 6. EC vs EO
# ─────────────────────────────────────────────────────────────────
print("\n[6/6] EC vs EO ...")
eceo_rows = []
for monkey in MONKEYS:
    for dt in DATES[monkey]["RS"]:
        arrays = sorted({int(a) for (m, d, a, c, b) in pca_store
                          if m == monkey and d == str(dt) and b == MAIN_BIN})
        for arr in arrays:
            if lib.get_scores(monkey, dt, arr, "EC", MAIN_BIN) is None:
                continue
            if lib.get_scores(monkey, dt, arr, "EO", MAIN_BIN) is None:
                continue
            pec = minpair_percentile(REAL_DF, CTRL_VEC, monkey, dt, arr, "EC", MAIN_BIN, MAX_DIM, NORM_MAIN)
            peo = minpair_percentile(REAL_DF, CTRL_VEC, monkey, dt, arr, "EO", MAIN_BIN, MAX_DIM, NORM_MAIN)
            eceo_rows.append(dict(monkey=monkey, date=dt, array=arr, pct_EC=pec, pct_EO=peo))
eceo_df = pd.DataFrame(eceo_rows)
eceo_df.to_csv(os.path.join(TABLE_DIR, "eceo_comparison.csv"), index=False)
n_both_sig = int(((eceo_df.pct_EC < SIG_THRESH) & (eceo_df.pct_EO < SIG_THRESH)).sum())
n_either_sig = int(((eceo_df.pct_EC < SIG_THRESH) | (eceo_df.pct_EO < SIG_THRESH)).sum())
n_ec_only = int(((eceo_df.pct_EC < SIG_THRESH) & ~(eceo_df.pct_EO < SIG_THRESH)).sum())
n_eo_only = int((~(eceo_df.pct_EC < SIG_THRESH) & (eceo_df.pct_EO < SIG_THRESH)).sum())
print(f"cells with both EC&EO computed: {len(eceo_df)}")
print(f"significant in both: {n_both_sig} | EC-only: {n_ec_only} | EO-only: {n_eo_only} | either: {n_either_sig}")


# ─────────────────────────────────────────────────────────────────
# Headline numbers -> JSON
# ─────────────────────────────────────────────────────────────────
headline = dict(
    main_bin=MAIN_BIN,
    norm_main=NORM_MAIN,
    n_cells_total=len(cellsdf) // 2,
    n_arrays=n_arrays,
    n_estimable_arrays=n_estimable,
    monkeys=MONKEYS,
    n_arrays_per_monkey={m: len(V1_ARRAYS[m]) for m in MONKEYS},
    min_oriented_significant=min_sig,
    p10_oriented_significant=p10_sig,
    n_qualifying_stability_arrays=len(qualifying),
    n_eceo_cells=len(eceo_df),
    n_eceo_both_sig=n_both_sig,
    n_eceo_ec_only=n_ec_only,
    n_eceo_eo_only=n_eo_only,
    sig_thresh=SIG_THRESH,
)
with open(os.path.join(TABLE_DIR, "headline_numbers.json"), "w") as f:
    json.dump(headline, f, indent=2)
print("\nheadline numbers:")
print(json.dumps(headline, indent=2))
