#!/bin/bash
#SBATCH -J train_dt_winert_wnoise_wsur
#SBATCH -n 1
#SBATCH -C "thin"
#SBATCH --gpus 1
#SBATCH -t 13:00:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./train_dt_wnoise_wsur_long1.out
#SBATCH -e ./train_dt_wnoise_wsur_long1.err
conda init
source  /home/x_yifji/.bashrc
conda activate /proj/gaia/RayDT/mamba_ray
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks
python train_dt.py --allTx True --wandb True --add_noise True --noise_sample 5 --env_index 1 --add_surface_index True