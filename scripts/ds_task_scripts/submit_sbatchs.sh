#!/bin/bash
cd /proj/gaia/RayDT/workspace/gfnrau/notebooks/scripts/ds_task_scripts/amend_hpsearch_scripts

# Loop to submit jobs from 4 to 20
for i in {63..92}
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