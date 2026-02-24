#!/bin/bash
#SBATCH --job-name=defame-Q4  # Name of the job
#SBATCH --output=defame_veritas_out_%j.txt  # Standard output file (%j = job ID)
#SBATCH --error=defame_veritas_err_%j.txt    # Standard error file (same as output if not specified)
#SBATCH --ntasks=1            # Number of tasks (usually 1 unless using MPI)
#SBATCH --cpus-per-task=10     # Number of CPU cores per task
#SBATCH --mem=150G                # Memory per node (8 GB, only for CPU part)
#SBATCH --gpus=1              # Number of GPUs required
#SBATCH --time=4-20:00:00         # Time limit (hh:mm:ss). If job runs longer than this, SLURM automatically kills it
#SBATCH --partition=all       # Partition to submit to (e.g., gpu, cpu)

#---------------------------------------------------------------
nvidia-smi  # Check GPU status before starting

python -c "import torch; print(torch.cuda.mem_get_info())"  # Check GPU memory info

export PATH="/mnt/vast/home/mk79honu/miniconda/bin:$PATH"
source ~/.bashrc  # Load the user's bashrc to ensure conda is initialized
conda activate defame
cd /mnt/vast/home/mk79honu/veritas_baselines/DEFAME  # Change to the project directory
CUDA_LAUNCH_BLOCKING=1 python -m scripts.run_config  # Run the evaluation script