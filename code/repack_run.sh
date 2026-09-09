#!/bin/bash
#SBATCH --job-name=repack
#SBATCH --mem=200G
#SBATCH -o output/repack_%A.out
#SBATCH -e output/repack_%A.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/maps_paper/code

# Repack the long-form binsweep errors into the compact pooled format.
python repack_binsweep.py