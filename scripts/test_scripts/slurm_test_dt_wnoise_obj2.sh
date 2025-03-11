#!/bin/bash
#SBATCH -J test_dt2_genz
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -t 9:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./test_dt_wnoise_long2_genz.out
#SBATCH -e ./test_dt_wnoise_long2_genz.err
#SBATCH --reservation=1g.10gb

conda init
source  /home/x_yifji/.bashrc
conda activate /proj/gaia/RayDT/mamba_ray
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks

# python test_dt.py --allTx True --add_noise True --noise_sample 5 --env_index 2 --eval_from 20240612-025537
python test_dt.py --allTx True --add_noise True --noise_sample 5 --env_index 2 --eval_from 20240612-025537 --test_data genz