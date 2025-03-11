#!/bin/bash
#SBATCH -J train_ac_winert_gym_cpu
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 12:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./slurm_train_ac_winert_gym.out
#SBATCH -e ./slurm_train_ac_winert_gym.err

micromamba activate /proj/raygnn/workspace/WINeRT/GFlowNet/mamba_gfn
cd /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/scripts
sh train_ac_winert_gym.sh

