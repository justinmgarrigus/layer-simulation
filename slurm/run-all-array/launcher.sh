#!/bin/bash 


######################
# === PARAMETERS === # 
######################

# How many nodes will be allocated. These nodes will not be allocated at the 
# exact same time. 
nodes=5

# How many tasks a single node should attempt to parallelize
tasks=3

# The maximum amount of time a single node will run for, in format "hh:mm:ss" 
duration="18:00:00"

# The queue the nodes will get added to
queue="normal"

# The path to the gpgpu-sim_distribution repository that will be sourced
sim_path=$WORK/gpgpu-sim_distribution_khoa

# "gpgpusim.config" file (will be copied by each thread during execution) 
sim_config_file=$sim_path/configs/SM70_TitanV_mesh_baseline/gpgpusim.config

# "config_*_islip.icnt" file (will be copied by each thread during execution)
islip_icnt_file=$sim_path/configs/SM70_TitanV_mesh_baseline/config_volta_islip.icnt

# The path to the singularity image that will be used 
singularity_path=$WORK/kldh-unt-gpgpusim_cuda9010v3.sif

# The path to where the trace files will be stored. Each thread will create
# their own directory within this one, which they will run the project in 
# reference towards. 
trace_path=$(pwd)/trace

# Which layers within the network will be run. These in reference to the order
# alexnet(5), resnet18(20), yolov5l(60), vgg16(13). For example, if 7 is
# passed as a layer, this cooresponds to layer 2 within resnet18. This must be
# a space-separated list of integer values. 
abs_layers=({1..98})

# Email address to send updates to (optional) 
email=


################
# === CODE === #
################

# Handle command file, which contains each layer to run on a new line.

> .commands

for index in "${abs_layers[@]}"; do 
    echo ${index} >> .commands
done 

# Remove the lock file 

find -name '.commands.*.*' -type f -delete
rm -f lines 
touch lines 


# Create .env file, which contains variables the slurm/worker scripts require

> .env 

echo "export MLNOC_GPGPUSIM_PATH=${sim_path}" >> .env
echo "export MLNOC_SINGULARITY_PATH=${singularity_path}" >> .env
echo "export MLNOC_TRACE_PATH=${trace_path}" >> .env 
echo "export MLNOC_CONFIG_FILE=${sim_config_file}" >> .env
echo "export MLNOC_ICNT_FILE=${islip_icnt_file}" >> .env


# Create output directory where job stdout/errors/trace go

mkdir -p output 
mkdir -p $trace_path


# Run SLURM script (other arguments are specified within the script) 

slurm_args=(-n ${tasks} -t ${duration} -p ${queue} -a 0-$((nodes-1)))
if [ -n "${email}" ]; then  
    slurm_args+=(--mail-type=all --mail-user=${email})
fi 

sbatch "${slurm_args[@]}" slurm.sh
