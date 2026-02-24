#!/bin/bash
#SBATCH --job-name=mocheg-gpt4o-baseline
#SBATCH --output=output_mocheg_baseline_%j.txt
#SBATCH --error=error_mocheg_baseline_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus=1
#SBATCH --time=0-03:00:00     # 3 hours for 150 samples
#SBATCH --partition=all

#---------------------------------------------------------------
# MOCHEG GPT-4o BASELINE Experiment
# Vanilla DEFAME with no blueprints
#---------------------------------------------------------------

echo "=========================================="
echo "SLURM Job: MOCHEG GPT-4o Baseline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo ""

# Setup environment
export PATH="/mnt/vast/home/mk79honu/miniconda/bin:$PATH"
source ~/.bashrc
conda activate defame
cd /mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME

# Run baseline experiment
python -m scripts.run_config /mnt/vast/home/mk79honu/DEFAME/config/experiments/mocheg_gpt_baseline.yaml
# python -m scripts.continue

echo ""
echo "Job completed at: $(date)"
echo "Results: out/mocheg_gpt_baseline/"