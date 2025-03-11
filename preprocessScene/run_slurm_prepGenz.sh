#!/bin/bash
#SBATCH --job-name=winert_genz_preprocess       # Set a name for your job
#SBATCH --nodes=1              # Request 2 nodes
#SBATCH --cpus-per-task=16       # Number of CPU cores per task
#SBATCH --gpus=1 
#SBATCH --gpus-per-task=1
#SBATCH --time=01:59:59         # Set the maximum walltime for your job (hh:mm:ss)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH --output=winert_genz_pp.out     # Define the name of the output log file
#SBATCH --error=winert_genz_pp.err       # Define the name of the error log file

# Activate the virtual environment
source /proj/raygnn/workspace/sionna_mamba/bin/activate

# Navigate to the directory where your script is located
cd /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/preprocessScene

# Run your script
python preprocessing_torch_cuda.py --testset genz

# Deactivate the virtual environment
deactivate