#!/bin/bash
#SBATCH -J ppalltest6
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 8:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/scripts/slurm_pp_less_repo/prev_logs_test/ppall_test_scene6.out
#SBATCH -e /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/scripts/slurm_pp_less_repo/prev_logs_test/ppall_test_scene6.err


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
python preprocessing_dt.py --env_index 6 --allTx True --Ofunc True --test_pp True
echo "Finished concurrent execution of tasks."
echo "Ending time: $(date)"
