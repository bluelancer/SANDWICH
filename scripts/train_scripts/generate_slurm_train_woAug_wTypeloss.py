import os
start_env_id = 1
end_env_id = 6
no_env_ids = [9,10,13,18,29,31,32,33,45,47,48,49,59,67,71,72,77,88,90,92,93,94,99]
template = """#!/bin/bash
#SBATCH -J train_dt_winert_across_rx_wonoise_wtype_thin_long_{env_id}
#SBATCH -n 1
#SBATCH -C "thin"
#SBATCH --gpus 1
#SBATCH -t 3:30:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./train_dt_winert_across_rx_wonoise_wtype_thin_long_{env_id}.out
#SBATCH -e ./train_dt_winert_across_rx_wonoise_wtype_thin_long_{env_id}.err

# Debugging start
echo "Starting time: $(date)"


# Initialize and activate conda environment
conda init
source  /home/x_yifji/.bashrc
conda activate /proj/gaia/RayDT/mamba_ray
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks
python train_dt.py --allTx True --wandb True --env_index {env_id} --add_type_loss True
echo "Finished concurrent execution of tasks."
echo "Ending time: $(date)"
"""

# Create directory for the SLURM scripts if it does not exist
output_dir = 'gen_train_scripts_woAug_wtype'
os.makedirs(output_dir, exist_ok=True)

# Generate SLURM scripts for obj ids from 1 to 100
for env_id in range(start_env_id,end_env_id):
    if env_id in no_env_ids:
        continue
    else:
        script_content = template.format(env_id=env_id)
        script_filename = os.path.join(output_dir, f'slurm_script_{env_id}.sh')
        
        with open(script_filename, 'w') as script_file:
            script_file.write(script_content)

print(f"Generated SLURM scripts for obj ids from {start_env_id} to {end_env_id} in the directory '{output_dir}'")
