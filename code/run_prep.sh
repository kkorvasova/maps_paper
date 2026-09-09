#!/bin/bash
#SBATCH --job-name=op_maps_prep
#SBATCH --mem=16G
#SBATCH -o output/op_maps_prep_%A.out
#SBATCH -e output/op_maps_prep_%A.err
#SBATCH --nodes=1

# One-time, non-parallel steps.
#   Usage:
#     sbatch run_prep.sh perms     # BEFORE the array job: build permutations
#     sbatch run_prep.sh merge     # AFTER  the array job: merge per-monkey output

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/maps_paper/code

STEP=${1:-perms}

if [ "$STEP" = "perms" ]; then
    python generate_permutations.py
elif [ "$STEP" = "merge" ]; then
    python compute_errors.py merge
else
    echo "unknown step: $STEP  (use 'perms' or 'merge')"
    exit 1
fi
