#!/bin/bash
#SBATCH -J train_dt_winert_wnoise_thin_long_95
#SBATCH -n 1
#SBATCH -C "thin"
#SBATCH --gpus 1
#SBATCH -t 12:30:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./train_dt_winert_across_rx_wnoise_thin_long_95.out
#SBATCH -e ./train_dt_winert_across_rx_wnoise_thin_long_95.err

# Debugging start
echo "Starting time: $(date)"


# Initialize and activate conda environment
conda init
source  /home/x_yifji/.bashrc
conda activate /proj/gaia/RayDT/mamba_ray
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks
python train_dt.py --allTx True --wandb True --add_noise True --noise_sample 5 --env_index 95
echo "Finished concurrent execution of tasks."
echo "Ending time: $(date)"
