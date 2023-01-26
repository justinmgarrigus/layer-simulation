#!/bin/bash

# single.sh: Runs the gemm executable on a batch of files. File batches are 
#   passed in either through: (1) standard input, or (2) command-line 
#   arguments. File batches must follow the format of "<weight input file> 
#   <x input file> <result output file> <error output file> 
#   <time output file>", where each file is a string and files are separated by 
#   one space.
# Execute: 
#   ./single.sh
#   ./single.sh <model_name> <layer_index>


if [ ! -d "/bin" ]
then
    echo "No bin directory exists! Run the project with 'python3 run.py'"
    exit 1
fi

# Processes a batch of files. Five parameters must be passed in: 
#   1: weight input file 
#   2: x input file 
#   3: result output file 
#   4: error output file 
#   5: time output file
function process_batch() {
    echo -n "weights: $1, x: $2, results: $3, errors: $4, time: $5 ... "

    if [ ! -f $1 ] || [ ! -f $2 ] 
    then 
        echo -e "\nError: weight file and input(x) file must both exist!" 
        exit 1 
    fi 
    
    mkdir -p $(dirname $3)
    mkdir -p $(dirname $4)
    mkdir -p $(dirname $5) 
    
    start_time=$(date +%s)
    ./build/gemm --w $1 --x $2 > $3 2> $4
    end_time=$(date +%s)
    duration=$((end_time-start_time))

    echo "Start: ${start_time}" > ${time_file}
    echo "End: ${end_time}" >> ${time_file}
    echo "Duration: ${duration}" >> ${time_file}

    echo "Done"
}

if [ $# -eq 2 ]
then 
    
    # User wants to only process one batch, where arguments are in the format
    # of "<model name> <layer number>" 
    
    range=$(find bin/$1 -name 'x_*' | wc -l) 
    if [ ${range} -eq 0 ]
    then 
        echo -n "Error: either no bin files exist, or the model name does not " 
        echo "match a directory specified in the 'bin' directory."
        exit 1 
    elif [ $2 -le 0 ] || [ $2 -gt ${range} ]
    then 
        echo "Error: index of $2 is out of range for model $1 (${range})" 
        exit 1 
    fi

    x_file=$(basename $(find bin/$1 -name 'x_*' | sort | sed -n $2p))
    layer_name=${x_file:2} # removes 'x_' from beginning
    layer_name="${layer_name%.*}" # removes file extension

    x_file="bin/$1/x_${layer_name}.bin" 
    weight_file="bin/$1/weight_${layer_name}.bin"
    result_file="output/$1/simResults_${layer_name}.txt"
    error_file="output/$1/simErrors_${layer_name}.txt" 
    time_file="output/$1/time_${layer_name}.txt" 
        
    process_batch      \
        ${weight_file} \
        ${x_file}      \
        ${result_file} \
        ${error_file}  \
        ${time_file}   

else 

    # Run in interactive mode, pulling from stdin. Lines inputted must follow
    # the format of: 
    #   <weight_file> <x_file> <result_file> <error_file> <time_file> 

    while read -r line
    do 
        parts=($line)
        process_batch ${parts[@]} 
    done
        
fi 
