#!/bin/bash
#SBATCH -J train_dt_winert_thin_long
#SBATCH -n 1
#SBATCH -C "thin"
#SBATCH --gpus 1
#SBATCH -t 6:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./train_dt_winert_across_rx_thin_long.out
#SBATCH -e ./train_dt_winert_across_rx_thin_long.err
conda init
source  /home/x_yifji/.bashrc
conda activate mamba_ray
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks
python train_decision_trans.py --Tx 1 --big_batch True --wandb True  --produce_fig True