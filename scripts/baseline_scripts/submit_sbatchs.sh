#!/bin/bash
cd /proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/scripts/baseline_scripts/gen_baseline_scripts_gpu

# Loop to submit jobs from 4 to 20
for i in {1..50}
do
  # Verify if slurm_script_${i}.sh exists
  if [ ! -f slurm_script_${i}.sh ]; then
    echo "slurm_script_${i}.sh does not exist"
    continue
  fi

  # Submit the job
  sbatch slurm_script_${i}.sh
  echo "Submitted slurm_script_${i}.sh"
done