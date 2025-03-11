import os

template = """#!/bin/bash
#SBATCH -J ppalltest{obj_id}
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 8:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/scripts/slurm_pp_less_repo/prev_logs_test/ppall_test_scene{obj_id}.out
#SBATCH -e /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/scripts/slurm_pp_less_repo/prev_logs_test/ppall_test_scene{obj_id}.err


# Debugging start
echo "Starting time: $(date)"


# Initialize and activate conda environment
conda init
source /home/x_yifji/.bashrc
conda activate /proj/raygnn/workspace/WINeRT/GFlowNet/mamba_gfn
export LD_LIBRARY_PATH=/proj/raygnn/workspace/WINeRT/GFlowNet/mamba_gfn/lib:$LD_LIBRARY_PATH

echo "Activated conda environment"

# Verify conda activation
echo "Conda environment:"
which python
python --version

cd /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks
echo "Current working directory: $(pwd)"

# Read the tasks from the tasks.txt file and execute them in parallel
echo "Starting concurrent execution of tasks..."
python preprocessing_dt.py --env_index {obj_id} --allTx True --Ofunc True --test_pp True
echo "Finished concurrent execution of tasks."
echo "Ending time: $(date)"
"""

# Create directory for the SLURM scripts if it does not exist
output_dir = 'slurm_scripts_test'
os.makedirs(output_dir, exist_ok=True)

# Generate SLURM scripts for obj ids from 1 to 100
for obj_id in range(1,11):
    script_content = template.format(obj_id=obj_id)
    script_filename = os.path.join(output_dir, f'slurm_script_{obj_id}_test.sh')
    
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)

print(f"Generated SLURM scripts for obj ids from 0 to 10 in the directory '{output_dir}'")
