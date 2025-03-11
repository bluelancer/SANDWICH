#!/bin/bash
#SBATCH -J train_dt_winert_across_rx_wonoise_wtype_thin_long_1
#SBATCH -n 1
#SBATCH -C "thin"
#SBATCH --gpus 1
#SBATCH -t 3:30:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./train_dt_winert_across_rx_wonoise_wtype_thin_long_1.out
#SBATCH -e ./train_dt_winert_across_rx_wonoise_wtype_thin_long_1.err

# Debugging start
echo "Starting time: $(date)"


# Initialize and activate conda environment
conda init
source  /home/x_yifji/.bashrc
conda activate /proj/gaia/RayDT/mamba_ray
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks
python train_dt.py --allTx True --wandb True --env_index 1 --add_type_loss True
echo "Finished concurrent execution of tasks."
echo "Ending time: $(date)"
