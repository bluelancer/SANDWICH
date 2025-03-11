import os
start_env_id = 1
end_env_id = 3
no_env_ids = [9,13,18,29,31,32,33,45,47,48,49,59,67,71,72,77,88,90,92,93,94,99]
template = """#!/bin/bash
#!/bin/bash
#SBATCH -J test_dt_genz_{env_id}
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -t 17:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./test_dt_genz_{env_id}_genz.out
#SBATCH -e ./test_dt_genz_{env_id}_genz.err
#SBATCH --reservation=1g.10gb

conda init
source  /home/x_yifji/.bashrc
conda activate /proj/gaia/RayDT/mamba_ray
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks

python test_dt.py --allTx True --add_noise True --noise_sample 5 --env_index {env_id} --eval_from latest --test_data genz
"""

# Create directory for the SLURM scripts if it does not exist

output_dir = 'Amend_scripts'

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
