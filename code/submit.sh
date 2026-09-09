#!/bin/bash
#SBATCH --job-name=op_maps
#SBATCH --mem=1000G
#SBATCH -o output/op_maps_%A.out
#SBATCH -e output/op_maps_%A.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/maps_paper/code

# 1. Build the 1000 permuted OP maps per array (must run first)
python generate_permutations.py

# 2. Main analysis: PCA + per-pair errors (real + permutations), save scores
python compute_errors.py