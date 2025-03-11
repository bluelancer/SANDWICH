#!/bin/bash
#SBATCH -J train_dt_winert
#SBATCH --gpus 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./train_dt_winert.out
#SBATCH -e ./train_dt_winert.err

conda activate /proj/raygnn/workspace/WINeRT/GFlowNet/mamba_gfn
cd /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/
python train_decision_trans.py --Tx 1 --Rx 1000 --optim_load True
python train_decision_trans.py --Tx 1 --Rx 1000 --optim_load True --add_noise True