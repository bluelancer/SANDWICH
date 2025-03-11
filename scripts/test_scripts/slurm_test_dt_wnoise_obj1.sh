#!/bin/bash
#SBATCH -J test_dt_genz1
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -t 18:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./test_dt_wnoise_long1_full.out
#SBATCH -e ./test_dt_wnoise_long1_full.err
#SBATCH --reservation=1g.10gb

conda init
source  /home/x_yifji/.bashrc
conda activate /proj/gaia/RayDT/mamba_ray
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks

python test_dt.py --allTx True --add_noise True --noise_sample 5 --env_index 1 --eval_from 20240611-203036 --test_data test
python test_dt.py --allTx True --add_noise True --noise_sample 5 --env_index 1 --eval_from 20240611-203036 --test_data genz