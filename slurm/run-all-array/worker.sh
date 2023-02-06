#!/bin/bash 

# worker.sh: launched from a SLURM batch job, this processes a single layer 
#   given its index. The index may be "relative" to a model, or it can be
#   "absolute" to the entire collection of models (see below). 
# Usage:
#   Singularity> ./worker.sh <model_name> <relative_index>
#       Runs the layer in the passed model cooresponding to the index, where 1
#       is the first layer, 2 is the second layer, ...
#   Singularity> ./worker.sh <absolute_index>
#       Interprets <array_index> as an absolute index, which is converted to a
#       value relative to the models; the model/layer count order is
#       alexnet(5), resnet18(20), yolov5l(60), vgg16(13), so an absolute index 
#       of 7 will become a relative index of 2 for the model resnet18.
# Requirements: This script requires that a variable "MLNOC_GPGPUSIM_PATH" be 
#   set. This must either be inside .bashrc, or it can exist within a local 
#   file named ".env". 


# We're running this from Singularity, so important variables may be unset 

source ~/.bashrc
if [ -f ".env" ]; then 
    source .env
fi

if [ -z "${MLNOC_GPGPUSIM_PATH+x}" ]; then 
    echo "Error: the environment variable MLNOC_GPGPUSIM_PATH must be" \
         "sourced in either the ~/bashrc file or a local .env file."
    exit 1 
fi


# Parses model and index from parameters

if [ $# -eq 2 ]; then 
    
    model=$1 
    index=$2

else 

    # Converts absolute index to relative index

    if [ $1 -lt 1 ] || [ $1 -gt 98 ]; then 
        echo "Error: layer index must be in range [1, 98]; received $1"
        exit 1
    elif [ $1 -lt 6 ]; then
        model="alexnet"
        index=$1
    elif [ $1 -lt 26 ]; then 
        model="resnet18"
        index=$(($1-5))
    elif [ $1 -lt 86 ]; then 
        model="yolov5l" 
        index=$(($1-25))
    else
        model="vgg16"
        index=$(($1-85))
    fi

fi  

set -- # clear passed arguments so sourced scripts dont get them.


# Find the base project directory where .git resides

if [ -n "$MLNOC_TRACE_PATH" ]; then
    exec_dir=$MLNOC_TRACE_PATH/$model/$index
    mkdir -p $exec_dir
else 
    exec_dir=$(pwd) 
fi 

while [ ! -d ".git" ] ; do 
    cd ..
done


# Set up simulator. This variable should be set prior to this script running

source ${MLNOC_GPGPUSIM_PATH}/setup_environment


# Copy configuration files to TODO 
cp $MLNOC_CONFIG_FILE $exec_dir
cp $MLNOC_ICNT_FILE $exec_dir


# Run our single layer

echo "Simulator path: ${MLNOC_GPGPUSIM_PATH}"  
echo "${model} ${index} started: $(date)"
script_path=$(readlink -f single.sh) 
cd $exec_dir
$script_path ${model} ${index}
echo "${model} ${index} done: $(date)"
