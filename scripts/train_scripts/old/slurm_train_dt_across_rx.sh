#!/bin/bash
#SBATCH -J train_dt_winert
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus 1
#SBATCH -t 3:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./train_dt_winert_across_rx.out
#SBATCH -e ./train_dt_winert_across_rx.err
conda init
source  /home/x_yifji/.bashrc
conda activate /proj/raygnn/workspace/WINeRT/GFlowNet/mamba_gfn
cd /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/
python train_decision_trans.py --Tx 1 --train_test_split True --produce_fig True
