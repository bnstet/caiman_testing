#!/bin/bash
#SBATCH --partition=fn_short
#SBATCH --job-name=caiman_pipeline
#SBATCH --mem=10000
#SBATCH --time=0-01:00:00
#SBATCH --tasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1

srun demo_slurm_subscript.sh
