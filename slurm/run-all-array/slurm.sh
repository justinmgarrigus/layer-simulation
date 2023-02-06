#!/bin/bash

# launcher.slurm: script to process each layer from GPGPU-Sim within 
#   Singularity. <ntasks> 


#SBATCH --output="output/%A_%a.out" 
#SBATCH --error="output/%A_%a.out" 


module load tacc-singularity


# Setup environment

for (( i=0; i<$SLURM_NTASKS; i++ )); do
    index=$((SLURM_ARRAY_TASK_ID * SLURM_NTASKS + i)) 
    srun -n 1 --exclusive ./thread.sh $index & 
done

wait
exit 0 


source .env
module load tacc-singularity


concurrent_job_count=0
while [ ${concurrent_job_count} -lt ${SLURM_NTASKS} ]; do
    
    # Operations involving locking will be done within these brackets

    {
        
        # It should not take longer than a second to finish a locked task, but
        # we're setting this to a high value to be sure. Anything longer than
        # 60 seconds either indicates system errors or deadlock. 
    
        flock -w 60 100 || { echo "Error: cannot acquire lock" ; exit 1 ; } 
        
        # Lock acquired, can now process file. Pop first line from file  
    
        task=$(head -n 1 .commands) 
        tail -n +2 .commands > .commands.tmp && mv .commands.tmp .commands
    
    } 100< .commands

    # We now have a task to process. 

done 




# num_tasks=0 
# while [ num_tasks -lt ${SLURM_NTASKS} ] && [ -f .commands ]; do
#     flock -w 60 .commands singularity
# done

# singularity exec --cleanenv $MLNOC_SINGULARITY_PATH worker.sh \
#     $SLURM_ARRAY_TASK_ID
