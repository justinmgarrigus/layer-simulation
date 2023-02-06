#!/bin/bash

# launcher.slurm: script to process each layer from GPGPU-Sim within 
#   Singularity. <ntasks> 


#SBATCH --output="output/%A_%a.out" 
#SBATCH --error="output/%A_%a.out" 


# Setup environment
module load tacc-singularity 

# Launch threads 
for (( i=0; i<$SLURM_NTASKS; i++ )); do
    index=$((SLURM_ARRAY_TASK_ID * SLURM_NTASKS + i)) 
    srun -n 1 --exclusive ./thread.sh $index & 
done

wait
