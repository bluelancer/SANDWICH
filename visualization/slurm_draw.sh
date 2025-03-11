#!/bin/bash
#SBATCH -J draw_1
#SBATCH -n 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64G
#SBATCH -t 1:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./draw_1.out
#SBATCH -e ./draw_1.err

conda init
source  /home/x_yifji/.bashrc
conda activate /proj/gaia/RayDT/mamba_ray
cd /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks
export LD_LIBRARY_PATH=/proj/gaia/RayDT/mamba_ray/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/proj/raygnn/workspace/WINeRT/GFlowNet/mamba_gfn/lib:$LD_LIBRARY_PATH
seq 1 99 | parallel -j 16 python draw_fig.py --allTx True --add_noise True --noise_sample 5 --env_index {}
