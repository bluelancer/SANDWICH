#!/bin/bash
#SBATCH -J train_dt_winert_wnoise
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus 1
#SBATCH -t 2-3:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./train_dt_winert_across_rx_wnoise.out
#SBATCH -e ./train_dt_winert_across_rx_wnoise.err
conda init
source  /home/x_yifji/.bashrc
conda activate /proj/raygnn/workspace/WINeRT/GFlowNet/mamba_gfn
cd /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/
python train_decision_trans.py --Tx 1 --add_noise True --train_test_split True --produce_fig True
